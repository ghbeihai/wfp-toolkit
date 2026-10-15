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
import sys


# 需要换算成每千字的按篇指标
PER_DOC_FEATURES = {
    "L1_SYN_004_question_sents": "问句/千字",
    "L1_SYN_004_rhet_sents": "修辞反问/千字",
}

# 字典型指标：展开成总和
DICT_FEATURES = {
    "L1_LEX_004_yqci_per_k": "语气词/千字(总和)",
    "L1_LEX_006_pron_per_k": "人称代词/千字(总和)",
}


def flatten_fp(fp: dict) -> dict:
    """把指纹拍平成标量字典，字典型展开成总和。"""
    chars = fp.get("chars", 0)
    result = {}

    for k, v in fp.items():
        if isinstance(v, dict):
            if k in DICT_FEATURES:
                # 字典型：展开成总和
                total = sum(x for x in v.values() if isinstance(x, (int, float)))
                result[DICT_FEATURES[k]] = round(total, 2)
        elif isinstance(v, (int, float)):
            if k in PER_DOC_FEATURES and chars > 0:
                # 按篇指标换算成每千字
                result[PER_DOC_FEATURES[k]] = round(v * 1000 / chars, 2)
            elif "_per_k" in k:
                result[k] = round(v, 2)
            elif k == "L1_SYN_001_avg_sl":
                result[k] = round(v, 1)
            elif k == "L1_LEX_001_sttr100":
                result[k] = round(v, 3)
    return result


def calc_delta(gen_norm: dict, baseline: dict) -> dict:
    """计算 z-score 距离。基线结构：{feature: {mean, sd, n}}"""
    bl = baseline.get("baseline", baseline)
    deltas = {}
    matched = []

    for k, gen_v in gen_norm.items():
        if k not in bl:
            # 按篇特征在基线里可能叫原名，尝试映射
            for bl_k, bl_v in bl.items():
                if bl_k in PER_DOC_FEATURES and k == PER_DOC_FEATURES[bl_k]:
                    # 基线是按篇的，用基线平均 chars 换算
                    mean_per_doc = bl_v.get("mean", 0)
                    # 估算平均 chars（从基线 n_docs 和 chars 推）
                    avg_chars = baseline.get("avg_chars", 500)
                    mean_per_k = mean_per_doc * 1000 / avg_chars
                    sd_per_doc = bl_v.get("sd", 0)
                    sd_per_k = sd_per_doc * 1000 / avg_chars
                    if sd_per_k > 0:
                        deltas[k] = abs(gen_v - mean_per_k) / sd_per_k
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
    ap.add_argument("--avg-chars", type=int, default=None, help="基线平均每篇字数（用于换算按篇指标）")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = ap.parse_args()

    gen = json.load(open(args.gen_fp, encoding="utf-8"))
    bl = json.load(open(args.baseline, encoding="utf-8"))
    avg_chars = args.avg_chars
    if avg_chars is None:
        # 从基线推算：用平均句长 × 估算平均句子数（每篇约 10-15 句）
        avg_sl = bl.get("baseline", {}).get("L1_SYN_001_avg_sl", {}).get("mean", 25)
        avg_chars = int(avg_sl * 12)  # 粗略估算每篇 12 句
        print(f"[warn] 未指定 --avg-chars，自动估算为 {avg_chars} 字/篇（基于平均句长 {avg_sl}），如不准请手动指定", file=__import__("sys").stderr)
    bl["avg_chars"] = avg_chars

    gen_norm = flatten_fp(gen)
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
