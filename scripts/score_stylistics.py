#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WFP · 文风指纹 — L3 软评分器（v0.4）

协议（与 AoT score_thinking 对齐）：
- 单维度独立调用；结构化 JSON {score: 1-7, evidence: [...], confidence: 0-1}；
- 多轮 temperature（config/llm.yaml temperatures，默认 0.3/0.7）取均值；证据逐字引用 ≤3 条；
- 断点续跑：已有有效分的 (doc, dim) 跳过；全部重试失败记 score=None/failed=True；
- 无 key 直接报错退出（无模拟模式）；429/5xx/截断/JSON 解析失败指数退避重试（llm_common）。

语料来源（三选一）：
  --corpus-dir <目录或文件>   目标作者语料（.txt/.md，或 SPI dict / AoT items JSON），doc_id = 文件名 stem
  --source spi [--spi-path]  data_sources.yaml 的 SPI 篇章 JSON（${SPI_ROOT}/${WFP_OBJ}）
  --source aot [--aot-dir]   data_sources.yaml 的 AoT 块目录

用法：
    python3 scripts/score_stylistics.py --corpus-dir ./原文 --out-dir ./l3_scores --l3-baseline-out l3_baseline.json
    python3 scripts/score_stylistics.py --corpus-dir ./原文 --dry-run     # 只估算调用次数
    python3 scripts/score_stylistics.py --source spi --pilot 2 14        # 抽样 2 篇 × 14 维
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import yaml

import data_sources
import llm_common

WFP_ROOT = Path(__file__).resolve().parent.parent
DIMS_PATH = WFP_ROOT / "config" / "l3_dimensions.yaml"
LLM_PATH = WFP_ROOT / "config" / "llm.yaml"
DEFAULT_OUT_DIR = WFP_ROOT / "data" / "results" / "l3_scores"

SYSTEM_PROMPT = """你是一位严谨的文风测量编码器（stylistic-pattern encoder）。

你会收到两部分输入：
1. 【文风维度定义】——某文风维度的五层编码：判定设问、本质、锚点词、文献依据、正反锚示例；
2. 【待分析文本】——一段自然语言文本。

你的任务：仅根据这段文本的实际内容，判断【文本作者】在该文风维度上的表现程度，
并输出严格 JSON（不要输出任何其他文字或解释）：
{"score": 1到7的整数, "evidence": ["原文片段1", ...最多3条], "confidence": 0到1之间的小数}

评分标尺：
- 7 = 文本强烈且一致地体现该维度（存在多处明确证据）
- 5~6 = 明显体现
- 3~4 = 部分或含糊体现
- 1~2 = 几乎不体现，或体现的是相反特征（按定义中写明的方向打分）

规则：
- 评分必须基于文本本身，不得臆测作者未表达的信息；
- evidence 必须逐字引用原文片段（保留原文语言，不做改写）；
- confidence 表示你对评分的把握程度（0=完全没把握，1=非常有把握）。"""


def render_definition(d: dict) -> str:
    lines = [
        f"维度名称：{d['name']}（{d.get('en', '')}）",
        f"判定设问：{d['question']}",
        f"本质：{d['essence']}",
        f"锚点词：{'、'.join(d.get('keywords', []))}",
        f"文献依据：{d['basis']}",
    ]
    anchors = d.get('anchors', {})
    for side, label in (('high', '高体现锚示例（典型体现）'), ('low', '低体现锚示例（相反体现）')):
        v = anchors.get(side)
        if v:
            lines.append(f"{label}：{v}")
    return "\n".join(lines)


class StylisticScorer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = llm_common.make_client(llm_common.require_key(WFP_ROOT), cfg)
        self.temperatures = list(cfg.get("temperatures") or [0.3, 0.7])
        self.evidence_max = int(cfg.get("evidence_max", 3))

    def score_once(self, definition: str, name: str, text: str) -> dict:
        text, truncated = llm_common.truncate(text, self.cfg)
        user = f"【文风维度定义】\n{definition}\n\n【待分析文本】\n{text}\n\n请按系统要求输出 JSON。"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        raw_scores: List[int] = []
        evidence: List[str] = []
        confidences: List[float] = []
        errors = []
        for t in self.temperatures:
            try:
                data = llm_common.chat_json(self.client, self.cfg, msgs, temperature=t, log_prefix=f"{name} t={t} ")
                score = int(data["score"])
                if not 1 <= score <= 7:
                    raise ValueError(f"score 越界: {score}")
            except Exception as ex:  # noqa: BLE001 —— 单轮失败记录后继续下一轮
                errors.append(f"t={t}: {type(ex).__name__}: {str(ex)[:120]}")
                print(f"    [warn] {name} t={t} 失败: {ex}", file=sys.stderr)
                continue
            raw_scores.append(score)
            for e in data.get("evidence", []) or []:
                e = str(e).strip()
                if e and e not in evidence and len(evidence) < self.evidence_max:
                    evidence.append(e)
            confidences.append(float(data.get("confidence", 0.5)))
        if not raw_scores:
            return {"score": None, "raw_scores": [], "evidence": [], "confidence": 0.0,
                    "simulated": False, "failed": True, "errors": errors, "truncated": truncated}
        return {
            "score": round(sum(raw_scores) / len(raw_scores), 2),
            "raw_scores": raw_scores,
            "evidence": evidence,
            "confidence": round(sum(confidences) / len(confidences), 2),
            "simulated": False,
            "truncated": truncated,
            "model": self.cfg.get("model"),
        }


def _items_from_json(path: Path, d) -> List[tuple]:
    out = []
    if isinstance(d, dict) and "items" in d:
        for it in d["items"]:
            txt = it.get("text") or it.get("content") or ""
            if txt.strip():
                out.append((str(it.get("id") or it.get("title") or path.stem), txt))
    elif isinstance(d, dict):
        for k, v in d.items():
            txt = " ".join(v) if isinstance(v, list) else str(v)
            if txt.strip():
                out.append((str(k), txt))
    return out


def load_corpus_dir(p: str) -> List[dict]:
    """目标作者语料：目录（.txt/.md/.json）或单文件 → [{id, text}]；doc_id 与 replicate.py 一致（文件 stem）。"""
    pp = Path(p)
    files = [pp] if pp.is_file() else sorted(f for f in pp.iterdir() if f.suffix.lower() in (".txt", ".md", ".json"))
    docs = []
    for f in files:
        if f.suffix.lower() == ".json":
            for i, t in _items_from_json(f, json.loads(f.read_text(encoding="utf-8"))):
                docs.append({"id": i, "text": t})
        else:
            t = f.read_text(encoding="utf-8")
            if t.strip():
                docs.append({"id": f.stem, "text": t})
            else:
                print(f"[warn] 跳过空文本 {f.name}", file=sys.stderr)
    return docs


def load_docs(args) -> List[dict]:
    if args.corpus_dir:
        docs = load_corpus_dir(args.corpus_dir)
    elif args.source == "spi":
        ds = data_sources.load()
        spi = Path(args.spi_path) if args.spi_path else data_sources.resolve(ds["corpus"]["spi_articles"], "SPI 篇章 JSON")
        docs = []
        for i, t in _items_from_json(spi, json.loads(spi.read_text(encoding="utf-8"))):
            if len(t) < 500:
                continue  # 跳过汇总说明条/空条（如"共收录 431 篇"条目）
            docs.append({"id": i, "text": t})
    else:
        ds = data_sources.load()
        aot = Path(args.aot_dir) if args.aot_dir else data_sources.resolve(ds["corpus"]["aot_dir"], "AoT 块目录")
        docs = []
        for fn in sorted(aot.glob("*.json")):
            for i, t in _items_from_json(fn, json.loads(fn.read_text(encoding="utf-8"))):
                docs.append({"id": f"{fn.stem}::{i}", "text": t})
    n = int(args.pilot[0]) if args.pilot else (args.limit or 0)
    return docs[:n] if n else docs


def out_name(doc_id: str) -> str:
    """输出文件名：可读前缀 + 短哈希，避免截断后重名覆盖。"""
    safe = re.sub(r"[^\w\u4e00-\u9fa5-]", "_", doc_id)[:40]
    return f"{safe}_{hashlib.md5(doc_id.encode('utf-8')).hexdigest()[:8]}.json"


def main():
    ap = argparse.ArgumentParser(description="WFP L3 软评分器 v0.4（需 DEEPSEEK_API_KEY）")
    ap.add_argument("--corpus-dir", help="目标作者语料目录或文件（.txt/.md/.json）；给了就忽略 --source")
    ap.add_argument("--source", choices=["spi", "aot"], default="spi", help="无 --corpus-dir 时的语料源（默认 spi）")
    ap.add_argument("--spi-path", help="覆盖 data_sources.yaml 的 SPI 篇章 JSON")
    ap.add_argument("--aot-dir", help="覆盖 data_sources.yaml 的 AoT 块目录")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"评分输出目录（默认 {DEFAULT_OUT_DIR}）")
    ap.add_argument("--l3-baseline-out", help="评分完成后汇总 L3 基线到此 JSON（等价于再跑 build_l3_baseline.py）")
    ap.add_argument("--pilot", nargs="*", default=None, help="抽样：'篇数 维度数'，如 --pilot 2 14")
    ap.add_argument("--limit", type=int, default=0, help="最多评多少篇（0=全量）")
    ap.add_argument("--dims", type=int, default=None, help="只用前 N 个维度")
    ap.add_argument("--dry-run", action="store_true", help="只打印待评篇数与调用次数估算，不调用 LLM")
    args = ap.parse_args()

    dims = yaml.safe_load(DIMS_PATH.read_text(encoding="utf-8"))["dimensions"]
    n_dims = args.dims or (int(args.pilot[1]) if args.pilot and len(args.pilot) > 1 else None)
    if n_dims:
        dims = dims[:n_dims]
    cfg = llm_common.load_llm_cfg(LLM_PATH)
    docs = load_docs(args)
    if not docs:
        sys.exit("错误：没有可评分的文本")
    out_dir = Path(args.out_dir)

    def build_tasks():
        for doc in docs:
            out_path = out_dir / out_name(doc["id"])
            existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
            for dim in dims:
                key = f"{doc['id']}::{dim['id']}"
                e = existing.get(key) or {}
                if (e.get("score") or 0) > 0 and not e.get("simulated") and not e.get("failed"):
                    continue
                yield doc, out_path, existing, key, dim

    tasks = list(build_tasks())
    n_temp = len(cfg.get("temperatures") or [0.3, 0.7])
    n_long = sum(1 for d in docs if len(d["text"]) > int(cfg.get("max_text_chars", 6000)))
    print(f"[info] {len(docs)} 篇 × {len(dims)} 维；待评 {len(tasks)} 项 × {n_temp} 轮 = {len(tasks) * n_temp} 次调用"
          f"（model={cfg.get('model')}，并发 {cfg.get('concurrency')}）；超长将截断 {n_long} 篇")
    if args.dry_run:
        return

    scorer = StylisticScorer(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, n_failed = 0, 0

    def run_task(t):
        doc, out_path, existing, key, dim = t
        res = scorer.score_once(render_definition(dim), dim["name"], doc["text"])
        res["dim"] = dim["id"]
        res["dim_name"] = dim["name"]
        res["doc_id"] = doc["id"]
        return out_path, existing, key, res

    with ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", 8))) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        for fut in as_completed(futs):
            out_path, existing, key, res = fut.result()
            existing[key] = res  # 只在主线程写文件
            out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            done += 1
            n_failed += bool(res.get("failed"))
            if done % 20 == 0 or done == len(tasks):
                print(f"  [prog] {done}/{len(tasks)} 项完成（失败 {n_failed}）", flush=True)

    print(f"[ok] L3 评分完成，结果在 {out_dir}（失败 {n_failed} 项，score=None 不进基线）")
    if args.l3_baseline_out:
        from build_l3_baseline import build_baseline, NoValidScores
        try:
            bl = build_baseline(out_dir)
        except NoValidScores as e:
            sys.exit(f"错误：{e}")
        Path(args.l3_baseline_out).write_text(json.dumps(bl, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] L3 基线：{bl['n_docs']} 篇 → {bl['n_dims']} 维 → {args.l3_baseline_out}")
    if n_failed:
        sys.exit(3)  # 有失败项：非 0 退出，提醒续跑


if __name__ == "__main__":
    main()
