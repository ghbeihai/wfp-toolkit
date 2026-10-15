#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每千字归一回测：把按篇指标换算成每千字后，再算 Delta。

解决篇幅差异陷阱：原文每篇 300 字 vs 生成文 1200 字时，按篇指标（问句/篇）
会被篇幅放大，导致 Delta 失真。统一换算成每千字后，对比才公平。
"""
import argparse
import json
from field_map import flatten, baseline_stat  # 唯一口径实现



def flatten_fp(fp: dict) -> dict:
    """拍平成 Delta 特征向量：统一走 field_map.flatten（含我:你截断、按篇→每千字）。"""
    return flatten(fp)


def calc_delta(gen_norm: dict, baseline: dict, avg_chars_override: int = None) -> dict:
    """计算 z-score 距离。gen_norm 已是 flatten 后的每千字口径，基线同口径（v1 旧基线由
    baseline_stat 兼容换算），因此这里不再二次换算。"""
    if avg_chars_override:
        baseline = dict(baseline, avg_chars=avg_chars_override)
    deltas = {}
    for k, gen_v in gen_norm.items():
        st = baseline_stat(baseline, k)
        if not st or not st[1]:
            continue
        deltas[k] = abs(gen_v - st[0]) / st[1]
    matched = list(deltas)
    avg = sum(deltas.values()) / len(matched) if matched else 0
    return {
        "delta": round(avg, 3),
        "n_features": len(matched),
        "matched_features": matched,
        "per_feature": {k: round(v, 2) for k, v in deltas.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="每千字归一回测")
    ap.add_argument("gen_fp", help="生成文指纹 JSON")
    ap.add_argument("--baseline", required=True, help="基线 JSON")
    ap.add_argument("--avg-chars", type=int, default=None,
                    help="基线平均每篇字数（用于换算按篇指标）")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = ap.parse_args()

    gen = json.load(open(args.gen_fp, encoding="utf-8"))
    bl = json.load(open(args.baseline, encoding="utf-8"))

    gen_norm = flatten_fp(gen)
    result = calc_delta(gen_norm, bl, args.avg_chars)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"每千字归一化后 Delta: {result['delta']}（{result['n_features']} 项）")
    print("\n逐项对比（z-score，越低越像）:")
    for k, v in sorted(result["per_feature"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
