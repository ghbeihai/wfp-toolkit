#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每千字归一回测：把按篇指标换算成每千字后，再算 Delta。

解决篇幅差异陷阱：原文每篇 300 字 vs 生成文 1200 字时，按篇指标（问句/篇）
会被篇幅放大，导致 Delta 失真。统一换算成每千字后，对比才公平。
"""
import argparse
import json
import sys


# 篇幅指标排除名单
EXCLUDE = {
    # 排除重复字段（L1 和 L4 重复的情感词）
    "L4_SUR_003_pos_per_k",
    "L4_SUR_003_neg_per_k",
    "chars", "sentences", "uniq_chars",
    "L1_LEX_001_ttr_char",
    "L2_DIS_001_avg_para_len",
}

# 按篇指标 → 换算成每千字
PER_DOC = {
    "L1_SYN_004_question_sents",
    "L1_SYN_004_rhet_sents",
}

# 字典型指标 → 展开成总和
DICT_FEATURES = {
    "L1_LEX_004_yqci_per_k": "语气词/千字(总和)",
    "L1_LEX_006_pron_per_k": "人称代词/千字(总和)",
    "L1_LEX_005_dm_per_k": "衔接词/千字(总和)",
}

# 展开后的标量字段（基线聚合时生成的 *_total）
DICT_TOTAL_FEATURES = {
    "L1_LEX_004_yqci_per_k_total",
    "L1_LEX_006_pron_per_k_total",
    "L1_LEX_005_dm_per_k_total",
}


def flatten_fp(fp: dict) -> dict:
    """我:你比例截断：限制在 0.1 - 10 之间，防止分母小爆炸"""
    if "L1_LEX_006_wo_ni_ratio" in fp:
        v = fp["L1_LEX_006_wo_ni_ratio"]
        fp["L1_LEX_006_wo_ni_ratio"] = max(0.1, min(v, 10.0))
    return _flatten_fp_impl(fp)

def _flatten_fp_impl(fp: dict) -> dict:
    """把指纹拍平成标量字典"""
    chars = fp.get("chars", 0)
    result = {}

    for k, v in fp.items():
        if k in EXCLUDE:
            continue
        if isinstance(v, dict):
            if k in DICT_FEATURES:
                total = sum(x for x in v.values() if isinstance(x, (int, float)))
                result[DICT_FEATURES[k]] = round(total, 2)
        elif isinstance(v, (int, float)):
            if k in PER_DOC and chars > 0:
                result[k] = round(v * 1000 / chars, 2)
            elif "_per_k" in k:
                result[k] = round(v, 2)
            elif k == "L1_SYN_001_avg_sl":
                result[k] = round(v, 1)
            elif k == "L1_LEX_001_sttr100":
                result[k] = round(v, 3)
            elif "ratio" in k:
                result[k] = round(v, 3)
    return result


def calc_delta(gen_norm: dict, baseline: dict, avg_chars_override: int = None) -> dict:
    """计算 z-score 距离"""
    bl = baseline.get("baseline", baseline)
    avg_chars = avg_chars_override or baseline.get("avg_chars", 500)

    deltas = {}
    matched = []

    for k, gen_v in gen_norm.items():
        if k not in bl:
            # 按篇特征在基线里可能叫原名
            for bl_k in PER_DOC:
                if k == bl_k:
                    bl_v = bl.get(bl_k, {})
                    mean = bl_v.get("mean", 0) * 1000 / avg_chars
                    sd = bl_v.get("sd", 0) * 1000 / avg_chars
                    if sd > 0:
                        deltas[k] = abs(gen_v - mean) / sd
                        matched.append(k)
                    break
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
            matched.append(k)

    total = sum(deltas.values())
    avg = total / len(matched) if matched else 0
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
    print(f"\n逐项对比（z-score，越低越像）:")
    for k, v in sorted(result["per_feature"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
