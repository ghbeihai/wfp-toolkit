#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每千字归一回测：把按篇指标换算成每千字后，再算 Delta。

解决篇幅差异陷阱：原文每篇 300 字 vs 生成文 1200 字时，按篇指标（问句/篇）
会被篇幅放大，导致 Delta 失真。统一换算成每千字后，对比才公平。

用法：
  python3 normalize_backtest.py <生成文指纹.json> --baseline <基线.json> [--json]

输出：归一化后的 Delta + 逐项对比表。
"""
import argparse
import json
import math
import sys


# 需要换算成每千字的按篇指标（key 前缀 → 换算公式）
PER_DOC_TO_PER_K = {
    "L1_SYN_004_question_sents": "问句/千字",
    "L1_SYN_004_rhet_sents": "修辞反问/千字",
}


def normalize_fp(fp: dict) -> dict:
    """把指纹统一成每千字归一化指标。"""
    chars = fp.get("chars", 0)
    if chars <= 0:
        return fp

    normalized = {}
    for k, v in fp.items():
        if not isinstance(v, (int, float)):
            continue
        if k in PER_DOC_TO_PER_K:
            normalized[k] = round(v * 1000 / chars, 2)
        elif "_per_k" in k:
            normalized[k] = round(v, 2)
        elif k == "L1_SYN_001_avg_sl":
            normalized[k] = round(v, 1)
        elif k == "L1_LEX_001_sttr100":
            normalized[k] = round(v, 3)
        elif "ratio" in k or "占比" in k:
            normalized[k] = round(v, 3)
    return normalized


def calc_delta(gen_norm: dict, baseline: dict) -> dict:
    """计算 z-score 距离。"""
    bl = baseline.get("baseline", baseline)
    deltas = {}
    common = 0
    for k, gen_v in gen_norm.items():
        if k not in bl:
            continue
        bl_v = bl[k]
        if isinstance(bl_v, dict):
            mean = bl_v.get("mean", 0)
            sd = bl_v.get("sd", 0)
        else:
            mean = bl_v
            sd = 0
        if sd > 0:
            deltas[k] = abs(gen_v - mean) / sd
            common += 1
    total = sum(deltas.values())
    avg = total / common if common else 0
    return {
        "delta": round(avg, 3),
        "n_features": common,
        "per_feature": {k: round(v, 2) for k, v in deltas.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="每千字归一回测")
    ap.add_argument("gen_fp", help="生成文指纹 JSON")
    ap.add_argument("--baseline", required=True, help="基线 JSON")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = ap.parse_args()

    gen = json.load(open(args.gen_fp, encoding="utf-8"))
    bl = json.load(open(args.baseline, encoding="utf-8"))

    gen_norm = normalize_fp(gen)
    result = calc_delta(gen_norm, bl)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"每千字归一化后 Delta: {result['delta']}（{result['n_features']} 项）")
    print(f"\n逐项对比（z-score，越低越像）:")
    for k, v in sorted(result["per_feature"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
