#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WFP · 文风指纹 — L3 软评分器（v0.3）

复用 AoT 已验证协议（<AoT_ROOT>/scripts/score_thinking.py）：
- 单维度独立调用，避免跨维度污染与顺序效应；
- 结构化输出 JSON {score: 1-7, evidence: [...], confidence: 0-1}；
- 双轮平均：temperature 0.3 / 0.7 两轮取均值，压低评分抖动；
- 证据句逐字引用原文 ≤3 条；
- 断点续跑：已完成的 (doc, dim) 跳过；
- 无 API key 时进入模拟模式（结果带 simulated 标记）。

维度尺子：config/l3_dimensions.yaml（14 维，判定设问 + 锚点词 + 正反锚）。
学术原则：维度定义有文献依据（docs/L3软评分维度清单.md）；anchors 为操作化示例。

用法：
    python3 scripts/score_stylistics.py --pilot 2 14     # 抽样 2 篇 × 14 维 × 双轮
    python3 scripts/score_stylistics.py --dims 4         # 只用前 4 维
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import yaml

WFP_ROOT = Path(__file__).resolve().parent.parent
DIMS_PATH = WFP_ROOT / "config" / "l3_dimensions.yaml"
LLM_PATH = WFP_ROOT / "config" / "llm.yaml"
# v0.4.20：数据源从 config/data_sources.yaml 读取
_DS = yaml.safe_load((WFP_ROOT / "config" / "data_sources.yaml").read_text(encoding="utf-8"))
AOT_ENV_PATH = Path(_DS["llm"]["env_path"])
SPI_PATH = Path(_DS["corpus"]["spi_articles"])
AOT_DIR = Path(_DS["corpus"]["aot_dir"])
OUT_DIR = WFP_ROOT / "data" / "results" / "l3_scores"

MAX_TEXT_CHARS = 6000  # 控制单次输入 token，超长截断

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


def load_key() -> Optional[str]:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k
    if AOT_ENV_PATH.exists():
        for line in AOT_ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class StylisticScorer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or load_key()
        if not self.api_key:
            print("错误：未检测到 DEEPSEEK_API_KEY。\n"
                  "请先 export DEEPSEEK_API_KEY=sk-... 或在 .env 文件中配置。",
                  file=sys.stderr)
            sys.exit(1)
        self.simulated = False
        try:
            from openai import OpenAI
        except ImportError:
            print("[warn] openai 未安装，尝试 pip install openai ...")
            os.system(f"{os.sys.executable} -m pip install -q openai")
            from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com", timeout=120)

    def _call_once(self, definition: str, name: str, text: str, temperature: float) -> dict:
        user = (
            f"【文风维度定义】\n{definition}\n\n"
            f"【待分析文本】\n{text}\n\n请按系统要求输出 JSON。"
        )
        resp = self.client.chat.completions.create(
            model="deepseek-chat",
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        return json.loads(content)

    def _simulate(self, name: str, definition: str, text: str) -> dict:
        tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
        hits = 0
        total = 0
        for w in re.findall(r"[\u4e00-\u9fff]{2,}", definition):
            total += 1
            if w in tokens:
                hits += 1
        score = 1 if total == 0 else max(1, min(7, round(hits / max(total, 1) * 50)))
        return {
            "score": score,
            "evidence": [text[:60] + ("…" if len(text) > 60 else "")],
            "confidence": 0.1,
            "simulated": True,
        }

    def score_once(self, definition: str, name: str, text: str) -> dict:
        raw_scores: List[int] = []
        evidence: List[str] = []
        confidences: List[float] = []
        simulated = False
        for t in (0.3, 0.7):
            for attempt in range(2):
                try:
                    data = self._call_once(definition, name, text, t) if not self.simulated else self._simulate(name, definition, text)
                    simulated = simulated or self.simulated or data.get("simulated", False)
                    score = int(data["score"])
                    if 1 <= score <= 7:
                        raw_scores.append(score)
                        ev = data.get("evidence", [])
                        for e in ev:
                            e = str(e).strip()
                            if e and e not in evidence and len(evidence) < 3:
                                evidence.append(e)
                        confidences.append(float(data.get("confidence", 0.5)))
                        break
                except Exception as ex:
                    if attempt == 1:
                        print(f"    [warn] {name} t={t} 失败: {ex}")
                    time.sleep(1.5)
        if not raw_scores:
            return {"score": 0.0, "raw_scores": [], "evidence": [], "confidence": 0.0, "simulated": True}
        return {
            "score": round(sum(raw_scores) / len(raw_scores), 2),
            "raw_scores": raw_scores,
            "evidence": evidence,
            "confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
            "simulated": simulated,
        }


def load_docs(source: str, limit: int) -> List[dict]:
    docs = []
    if source == "spi":
        d = json.loads(SPI_PATH.read_text(encoding="utf-8"))
        for title, v in d.items():
            txt = " ".join(v) if isinstance(v, list) else str(v)
            if len(txt) < 500:
                continue  # 跳过汇总说明条/空条（如"共收录 431 篇"条目）
            docs.append({"id": title, "text": txt[:MAX_TEXT_CHARS]})
            if limit and len(docs) >= limit:
                break
    else:
        for fn in sorted(AOT_DIR.glob("*.json")):
            d = json.loads(fn.read_text(encoding="utf-8"))
            for it in d.get("items", []):
                txt = it.get("text") or it.get("content") or ""
                if not txt:
                    continue
                docs.append({"id": f"{fn.stem}::{it.get('id','')}", "text": txt[:MAX_TEXT_CHARS]})
                if limit and len(docs) >= limit:
                    break
            if limit and len(docs) >= limit:
                break
    return docs


def main():
    ap = argparse.ArgumentParser(description="WFP L3 软评分器 v0.3")
    ap.add_argument("--pilot", nargs="*", default=None, help="抽样：'篇数 维度数'，如 --pilot 2 14")
    ap.add_argument("--source", choices=["spi", "aot"], default="spi", help="语料源（默认 SPI 篇章级）")
    ap.add_argument("--dims", type=int, default=None, help="只用前 N 个维度")
    args = ap.parse_args()

    dims = yaml.safe_load(DIMS_PATH.read_text(encoding="utf-8"))["dimensions"]
    if args.dims:
        dims = dims[: args.dims]
    llm_cfg = yaml.safe_load(LLM_PATH.read_text(encoding="utf-8"))

    scorer = StylisticScorer()
    if scorer.simulated:
        print("[!] 未找到 DEEPSEEK_API_KEY，进入模拟模式（结果带 simulated 标记）")

    n_docs = int(args.pilot[0]) if args.pilot else 0  # 0=全量
    docs = load_docs(args.source, n_docs)
    print(f"[info] 待评 {len(docs)} 篇 × {len(dims)} 维 × 双轮 = {len(docs)*len(dims)*2} 次调用")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(docs) * len(dims) * 2
    done = 0

    def build_tasks():
        for doc in docs:
            out_path = OUT_DIR / f"{doc['id'][:40].replace('/', '_')}.json"
            existing = {}
            if out_path.exists():
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            for dim in dims:
                key = f"{doc['id']}::{dim['id']}"
                if key in existing and existing[key].get("score", 0) > 0 and not existing[key].get("simulated", False):
                    continue
                yield doc, out_path, existing, key, dim

    tasks = list(build_tasks())

    def run_task(t):
        doc, out_path, existing, key, dim = t
        definition = render_definition(dim)
        res = scorer.score_once(definition, dim["name"], doc["text"])
        res["dim"] = dim["id"]
        res["dim_name"] = dim["name"]
        return out_path, existing, key, res

    with ThreadPoolExecutor(max_workers=llm_cfg.get("concurrency", 12)) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                out_path, existing, key, res = fut.result()
                existing[key] = res
                out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                done += 2
                if done % 20 == 0 or done == total:
                    print(f"  [prog] {done}/{total} 次调用完成", flush=True)
            except Exception as ex:
                print(f"  [err] {key} 失败: {ex}")

    print(f"[ok] L3 评分完成，结果在 {OUT_DIR}")


if __name__ == "__main__":
    main()
