#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 LLM 工具：key 加载 / 配置 / 带退避重试的 JSON 调用。

wfp-toolkit、aot-toolkit、spi-toolkit 各带一份相同实现（三个仓库互不 import；源头 wfp-toolkit/scripts/llm_common.py）。

key 查找顺序（第一个命中即用，load_key 返回 (key, 来源)）：
  1. 环境变量 DEEPSEEK_API_KEY
  2. 环境变量 WENBI_ENV_FILE 指向的 .env
  3. 本仓库根目录 .env
  4. 兄弟仓库 .env：$AOT_ROOT/.env、$SPI_ROOT/.env、$WFP_ROOT/.env
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

DEFAULTS = {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "temperatures": [0.3, 0.7],
    "concurrency": 8,
    "max_retries": 5,
    "backoff_base": 2.0,   # 首次重试等待秒数，之后翻倍
    "backoff_max": 60.0,
    "timeout": 120,
    "max_text_chars": 6000,
}


def _read_env_file(path: Path) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return v or None
    except OSError as e:
        print(f"[warn] 读取 {path} 失败: {e}", file=sys.stderr)
    return None


def load_key(repo_root: Path | str | None = None):
    """返回 (key, 来源描述)；找不到返回 (None, 已尝试列表)。"""
    k = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if k:
        return k, "env:DEEPSEEK_API_KEY"
    cands = []
    if os.environ.get("WENBI_ENV_FILE"):
        cands.append(Path(os.environ["WENBI_ENV_FILE"]))
    if repo_root:
        cands.append(Path(repo_root) / ".env")
    for var in ("AOT_ROOT", "SPI_ROOT", "WFP_ROOT"):
        if os.environ.get(var):
            cands.append(Path(os.environ[var]) / ".env")
    seen = []
    for p in cands:
        if p in seen:
            continue
        seen.append(p)
        if p.exists():
            v = _read_env_file(p)
            if v:
                return v, str(p)
    return None, "tried: env DEEPSEEK_API_KEY, " + ", ".join(map(str, seen))


def require_key(repo_root=None) -> str:
    """找不到 key 时打印清晰错误并 exit(1)。WENBI_MOCK_LLM 打开时返回占位 key（测试用）。"""
    if os.environ.get("WENBI_MOCK_LLM"):
        return "mock"
    key, src = load_key(repo_root)
    if not key:
        print("错误：未检测到 DEEPSEEK_API_KEY。\n"
              "请 export DEEPSEEK_API_KEY=sk-...，或写入 .env（" + src + "）。", file=sys.stderr)
        sys.exit(1)
    return key


def load_llm_cfg(path: Path | str | None, section: str | None = None) -> dict:
    """读 YAML（可指定子节，如 SPI models.yaml 的 llm:），缺省键用 DEFAULTS 补齐。"""
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        import yaml
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if section:
            user = user.get(section) or {}
        if "timeout" not in user and "timeout_seconds" in user:
            user["timeout"] = user["timeout_seconds"]
        # 兼容旧键：rounds: [0.3, 0.7] 视为 temperatures
        if "temperatures" not in user and isinstance(user.get("rounds"), list):
            user["temperatures"] = user["rounds"]
        cfg.update({k: v for k, v in user.items() if v is not None})
    return cfg


class RateLimitError(Exception):
    """mock 用：类名与 openai.RateLimitError 一致，走同一重试分支。"""
    status_code = 429


class _MockClient:
    """离线 mock（WENBI_MOCK_LLM=1；=flaky 时每个请求首次抛 429）。按 prompt 关键词返回确定性 JSON。"""

    def __init__(self, mode: str):
        self.mode = mode
        self.calls = 0
        self._seen = set()
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        text = json.dumps(kw.get("messages", []), ensure_ascii=False)
        key = hash(text) ^ hash(kw.get("temperature"))
        if self.mode == "flaky" and key not in self._seen:
            self._seen.add(key)
            raise RateLimitError("mock 429")
        if not kw.get("response_format"):  # 文本生成（闭环生成用）
            tpl = ["我们今天聊一聊营养这件事。其实很多人都有误区，对吧？",
                   "营养不是越多越好，关键在于平衡，在于长期坚持，在于对自己身体的了解和尊重。",
                   "吃饭要慢。蔬菜要多。别熬夜。你记住了吗？"][len(text) % 3]  # 随 prompt 变化，模拟改写效果
            body = (tpl * 20 + "\n") * 3
            out = None
        elif "体裁分类器" in text:
            out = {"genre": "essay", "confidence": 0.7}
        elif "篇章框架" in text or "棱角保留任务" in text or "思维大纲" in text or "SPI+AoT 先导" in text:
            out = {"stance": "别指望吃一样东西就把问题解决，先把胡说八道撇开",
                   "thesis_arc": "开题硬砸错误直觉→中段拆真正相关的点→纠偏极端忌口→收束钉死：辅助不是主力",
                   "sections": [
                     {"role": "破题", "purpose": "砸神效幻想", "point": "没那回事", "move": "转到真正机制"},
                     {"role": "展开", "purpose": "给可操作主见", "point": "原料要够但工厂得开工", "move": "纠偏"},
                     {"role": "收束", "purpose": "钉死边界", "point": "辅助不是主力", "move": ""}],
                   "sharp_claims": ["神效食物是胡扯", "均衡比猛补重要"],
                   "must_include": ["食物优先", "别指望神药"],
                   "must_avoid": ["中庸两边对", "流水账罗列"],
                   "no_smoothing": ["不许改成也许都有道理"],
                   "closing": "硬收：听医生，别押在吃上",
                   "spi_used": ["Applied Life Domains"],
                   "aot_used": ["抽象-具体转换"],
                   "outline_text": "先砸「吃黑补黑」；再讲微量元素与蛋白是原料不是仙丹；最后钉死营养只是辅助。"}
        elif "棱角保留" in text:
            out = {"score": 6, "confidence": 0.7, "kept_edges": ["硬砸神效"], "smoothed": [],
                   "fix_hints": ["收束再硬半句"], "summary": "mock 棱角尚可"}
        elif "思维依从" in text or "SPI+AoT 依从" in text:
            out = {"score": 5, "structure_score": 5, "confidence": 0.7, "hits": ["有破题"], "misses": ["收束偏软"],
                   "fix_hints": ["按节推进，收束回到立场句"], "summary": "mock 依从中等"}
        elif "作者鉴别" in text:
            out = {"choice": "A", "confidence": 0.6, "reason": "mock"}
        elif "篇章框架" in text or "棱角保留任务" in text or "思维大纲" in text or "SPI+AoT 先导" in text:
            out = {"stance": "别指望吃一样东西就把问题解决，先把胡说八道撇开",
                   "thesis_arc": "开题硬砸错误直觉→中段拆真正相关的点→纠偏极端忌口→收束钉死：辅助不是主力",
                   "sections": [
                     {"role": "破题", "purpose": "砸神效幻想", "point": "没那回事", "move": "转到真正机制"},
                     {"role": "展开", "purpose": "给可操作主见", "point": "原料要够但工厂得开工", "move": "纠偏"},
                     {"role": "收束", "purpose": "钉死边界", "point": "辅助不是主力", "move": ""}],
                   "sharp_claims": ["神效食物是胡扯", "均衡比猛补重要"],
                   "must_include": ["食物优先", "别指望神药"],
                   "must_avoid": ["中庸两边对", "流水账罗列"],
                   "no_smoothing": ["不许改成也许都有道理"],
                   "closing": "硬收：听医生，别押在吃上",
                   "spi_used": ["Applied Life Domains"],
                   "aot_used": ["抽象-具体转换"],
                   "outline_text": "先砸「吃黑补黑」；再讲微量元素与蛋白是原料不是仙丹；最后钉死营养只是辅助。"}
        elif "metaphors" in text:
            out = {"metaphors": [{"text": "人生是一场旅程", "domain": "旅程", "explain": "mock"}]}
        elif "cluster_id" in text:
            out = {"hits": [{"cluster_id": 1, "relevance": 6, "reason": "mock"},
                            {"cluster_id": 3, "relevance": 4, "reason": "mock"}]}
        elif '"anchors"' in text:
            out = {"anchors": {"low": "mock", "high": "mock"}}
        else:
            out = {"score": 4 + (len(text) % 3), "evidence": ["mock"], "confidence": 0.6}

        content_ = body if out is None else json.dumps(out, ensure_ascii=False)

        class _M:
            content = content_

        class _C:
            message = _M()
            finish_reason = "stop"

        class _R:
            choices = [_C()]
        return _R()


def make_client(key: str, cfg: dict):
    mode = os.environ.get("WENBI_MOCK_LLM")
    if mode:
        return _MockClient(mode)
    try:
        from openai import OpenAI
    except ImportError:
        print("错误：缺少 openai 包（pip install openai）。", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=key, base_url=cfg.get("base_url", DEFAULTS["base_url"]),
                  timeout=cfg.get("timeout", DEFAULTS["timeout"]))


def _is_retryable(e: Exception) -> bool:
    name = type(e).__name__
    if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
                "JSONDecodeError", "TruncatedOutput"):
        return True
    status = getattr(e, "status_code", None)
    return status in (408, 409, 429, 500, 502, 503, 504)


class TruncatedOutput(Exception):
    pass


# ---------- 内容哈希缓存 ----------
# 相同（模型, base_url, messages, temperature, max_tokens, json_mode）的调用直接复用结果：
# 文本没变就不重算（SPI/AoT/WFP 打分、隐喻、体裁、盲测都走这里）。
#   WENBI_CACHE_DIR 缓存目录（默认 ~/.cache/wenbi/llm）；WENBI_NO_CACHE=1 关闭。
#   mock 模式与真实调用分开缓存（键里含 mock 标记）。
def _cache_dir():
    if os.environ.get("WENBI_NO_CACHE"):
        return None
    d = Path(os.environ.get("WENBI_CACHE_DIR") or (Path.home() / ".cache" / "wenbi")) / "llm"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def _cache_key(kw: dict, cfg: dict) -> str:
    import hashlib
    blob = json.dumps({"k": kw, "base_url": cfg.get("base_url"), "mock": os.environ.get("WENBI_MOCK_LLM") or ""},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key):
    d = _cache_dir()
    p = d / key[:2] / f"{key}.json" if d else None
    if p and p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["content"]
        except (OSError, ValueError, KeyError):
            return None
    return None


def _cache_put(key, content: str):
    d = _cache_dir()
    if not d:
        return
    p = d / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)  # 原子写，并发安全


CACHE_STATS = {"hit": 0, "miss": 0}


def chat_text(client, cfg: dict, messages: list, temperature: float | None = None,
              max_tokens: int | None = None, log_prefix: str = "") -> str:
    """纯文本生成（闭环生成用）；同样带退避重试与缓存。"""
    return chat_json(client, cfg, messages, temperature=temperature, max_tokens=max_tokens,
                     json_mode=False, log_prefix=log_prefix)


def chat_json(client, cfg: dict, messages: list, temperature: float | None = None,
              max_tokens: int | None = None, json_mode: bool = True, log_prefix: str = ""):
    """调用 chat.completions 并解析 JSON；对 429/5xx/超时/截断/JSON 解析失败做指数退避重试。
    最终失败抛出最后一次异常。返回 dict。"""
    retries = int(cfg.get("max_retries", DEFAULTS["max_retries"]))
    base = float(os.environ.get("WENBI_LLM_BACKOFF") or cfg.get("backoff_base", DEFAULTS["backoff_base"]))
    cap = float(cfg.get("backoff_max", DEFAULTS["backoff_max"]))
    kw = dict(model=cfg.get("model", DEFAULTS["model"]), messages=messages)
    if temperature is not None:
        kw["temperature"] = temperature
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    ckey = _cache_key(kw, cfg)
    cached = _cache_get(ckey)
    if cached is not None:
        CACHE_STATS["hit"] += 1
        return json.loads(cached) if json_mode else cached
    CACHE_STATS["miss"] += 1
    last = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(**kw)
            ch = resp.choices[0]
            content = (ch.message.content or "").strip()
            if getattr(ch, "finish_reason", None) == "length":
                raise TruncatedOutput("LLM 输出被截断（finish_reason=length）")
            if not json_mode:
                _cache_put(ckey, content)
                return content
            if content.startswith("```"):
                content = content.strip("`")
                content = content[4:] if content.lower().startswith("json") else content
            data = json.loads(content)
            _cache_put(ckey, content)  # 只缓存成功解析的结果
            return data
        except Exception as e:  # noqa: BLE001 —— 分类后决定是否重试
            last = e
            if attempt >= retries or not _is_retryable(e):
                raise
            delay = min(cap, base * (2 ** attempt)) * (0.5 + random.random())  # 指数退避 + 抖动
            print(f"[retry] {log_prefix}{type(e).__name__}: {str(e)[:80]}；{delay:.1f}s 后重试 "
                  f"({attempt + 1}/{retries})", file=sys.stderr)
            time.sleep(delay)
    raise last  # pragma: no cover


def truncate(text: str, cfg: dict, label: str = "") -> tuple[str, bool]:
    """按 max_text_chars 截断；截断时 stderr 提示并返回 truncated=True。"""
    n = int(cfg.get("max_text_chars", DEFAULTS["max_text_chars"]))
    if n and len(text) > n:
        print(f"[warn] {label}文本 {len(text)} 字 > {n}，仅取前 {n} 字评分（结果带 truncated 标记）",
              file=sys.stderr)
        return text[:n], True
    return text, False


def score_1to7(client, cfg: dict, messages: list, label: str = "", evidence_max: int = 3,
               truncated: bool = False) -> dict:
    """多温度（cfg.temperatures）各调用一次取均值的 1-7 打分协议；单轮失败记录后继续，
    全部失败返回 score=None/failed=True（不伪装成 0 分）。"""
    scores, confs, evs, errors = [], [], [], []
    for t in list(cfg.get("temperatures") or DEFAULTS["temperatures"]):
        try:
            data = chat_json(client, cfg, messages, temperature=t, log_prefix=f"{label} t={t} ")
            s = int(data["score"])
            if not 1 <= s <= 7:
                raise ValueError(f"score 越界: {s}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"t={t}: {type(e).__name__}: {str(e)[:120]}")
            print(f"    [warn] {label} t={t} 失败: {e}", file=sys.stderr)
            continue
        scores.append(s)
        confs.append(float(data.get("confidence", 0.5)))
        for e in data.get("evidence", []) or []:
            e = str(e).strip()
            if e and e not in evs and len(evs) < evidence_max:
                evs.append(e)
    if not scores:
        return {"score": None, "raw_scores": [], "evidence": [], "confidence": 0.0,
                "simulated": False, "failed": True, "errors": errors, "truncated": truncated}
    return {"score": round(sum(scores) / len(scores), 2), "raw_scores": scores, "evidence": evs,
            "confidence": round(sum(confs) / len(confs), 2), "simulated": False,
            "truncated": truncated, "model": cfg.get("model")}
