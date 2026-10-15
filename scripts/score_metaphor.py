#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4 隐喻识别（DeepSeek 版）

用法：
  python3 score_metaphor.py 某篇文章.txt --json
"""
import argparse

import llm_common
import json
import re
import sys
from pathlib import Path

METAPHOR_DOMAINS = {
    "旅程": "把人生/事业/关系比作旅程",
    "战争": "把争论/竞争/对抗比作战争",
    "建筑": "把体系/人格/作品比作建筑",
    "机器": "把组织/身体/社会比作机器",
    "生命": "把抽象事物比作生物",
    "市场": "把关系/能力/事物比作商品交易",
    "游戏": "把生活/竞争比作游戏博弈",
    "剧场": "把社会/人生比作舞台戏剧",
    "容器": "把抽象状态比作容器",
    "光暗": "把情绪/认知比作光暗",
    "上下": "把地位/数量/情绪比作垂直方位",
    "温度": "把关系/情绪比作温度",
}

SYSTEM_PROMPT = """你是一位严谨的隐喻识别编码员。

你会收到一段中文文本。任务：通读全文，找出所有【概念隐喻】——即把抽象事物比作具体事物的表达。

输出严格 JSON，无其他文字：
{
  "metaphors": [
    {"text": "原文片段（逐字引用）", "domain": "12根域之一", "explain": "为什么是这个域，20字内"}
  ]
}

12 个根域只能是：旅程、战争、建筑、机器、生命、市场、游戏、剧场、容器、光暗、上下、温度。
找不到任何隐喻时输出 {"metaphors": []}。"""

CHUNK_SIZE = 2000  # 分块阈值：超过 2000 字就分块


def count_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _split_long(p: str, max_len: int) -> list[str]:
    """单段超过 max_len：先按句末标点切，再不行就硬切。"""
    if len(p) <= max_len:
        return [p]
    sents = [x for x in re.split(r"(?<=[。！？!?；;…])", p) if x]
    out, cur = [], ""
    for s_ in sents:
        while len(s_) > max_len:  # 单句也超长：硬切
            if cur:
                out.append(cur)
                cur = ""
            out.append(s_[:max_len])
            s_ = s_[max_len:]
        if len(cur) + len(s_) > max_len and cur:
            out.append(cur)
            cur = s_
        else:
            cur += s_
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_len: int = CHUNK_SIZE) -> list[str]:
    """按段落分块，每块不超过 max_len 字；单段超长时按句切分（原实现会把整段塞进一个超大块）。"""
    paragraphs = [q for p in text.split("\n") if p.strip() for q in _split_long(p.strip(), max_len)]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > max_len and current:
            chunks.append(current)
            current = p
        else:
            current = current + "\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


def score_chunk(text: str, client, cfg: dict, model: str | None = None):
    """返回该块的隐喻列表；429/5xx/截断/JSON 失败由 llm_common 指数退避重试，
    仍失败返回 None（调用方计入 failed_chunks，而不是静默当作 0 个隐喻）。"""
    c = dict(cfg)
    if model:
        c["model"] = model
    try:
        data = llm_common.chat_json(
            client, c,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            temperature=float(cfg.get("metaphor_temperature", 0.0)),
            max_tokens=int(cfg.get("metaphor_max_tokens", 8192)), log_prefix="metaphor ")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 分块失败: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None
    return data.get("metaphors", []) if isinstance(data, dict) else None


def main():
    ap = argparse.ArgumentParser(description="L4 隐喻识别（DeepSeek 版；模型/温度来自 config/llm.yaml）")
    ap.add_argument("file", help="待分析文本文件")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--out", help="把 JSON 结果写入此文件（供 merge/persona 引用）")
    ap.add_argument("--model", default=None, help="覆盖 config/llm.yaml 的 model")
    args = ap.parse_args()

    text = open(args.file, encoding="utf-8").read()
    chars = count_chars(text)
    if chars == 0:
        sys.exit(f"错误：{args.file} 是空文本")
    cfg = llm_common.load_llm_cfg(Path(__file__).resolve().parent.parent / "config" / "llm.yaml")
    client = llm_common.make_client(llm_common.require_key(Path(__file__).resolve().parent.parent), cfg)

    # 分块打分
    chunks = split_text(text)
    all_metaphors = []
    failed_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  分块 {i+1}/{len(chunks)}（{len(chunk)} 字）...", file=sys.stderr)
        got = score_chunk(chunk, client, cfg, args.model)
        if got is None:
            failed_chunks.append(i)
        else:
            all_metaphors.extend(got)
    if failed_chunks:
        print(f"[warn] {len(failed_chunks)}/{len(chunks)} 个分块失败，结果偏低：{failed_chunks}", file=sys.stderr)

    # 按根域聚合
    domain_counts = {d: 0 for d in METAPHOR_DOMAINS}
    unknown = {}
    for m in all_metaphors:
        d = m.get("domain", "")
        if d in domain_counts:
            domain_counts[d] += 1
        else:
            unknown[d] = unknown.get(d, 0) + 1

    total = len(all_metaphors)
    result = {
        "chars": chars,
        "total_metaphors": total,
        "meta_per_k": round(total / max(chars, 1) * 1000, 3),
        "domain_counts": domain_counts,
        "unknown_domains": unknown,
        "domain_per_k": {d: round(n / max(chars, 1) * 1000, 3) for d, n in domain_counts.items()},
        "failed_chunks": failed_chunks,
        "n_chunks": len(chunks),
        "examples": [{"text": m.get("text", "")[:60], "domain": m.get("domain", "")} for m in all_metaphors[:10]],
    }

    if unknown:
        print(f"[warn] {sum(unknown.values())} 处隐喻的根域不在 12 域内，未计入密度：{unknown}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] 写出 {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"总隐喻数：{total}（每千字 {result['meta_per_k']}）")
        print("各根域密度（每千字）：")
        for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
            if n > 0:
                print(f"  {d}: {n} 处 ({result['domain_per_k'][d]}/千字)")


if __name__ == "__main__":
    main()
