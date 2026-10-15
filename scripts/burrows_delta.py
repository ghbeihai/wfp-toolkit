#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Burrows Delta：基于高频虚词词频分布的作者识别指标。

原理：不同作者对"的/了/是/在/和/就/都/而/及/与/这/那/有/不"等高频虚词
的使用频率有稳定差异，这种差异比句长/问句等通用指标更能区分作者。

用法：
  python3 burrows_delta.py <文本1.txt> <文本2.txt> [--top 50]
  python3 burrows_delta.py --corpus <目录> --baseline <基线.json>
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


# 中文高频功能词（虚词为主，按常见度排序）
COMMON_FUNCTION_WORDS = [
    # 助词
    "的", "了", "着", "过",
    # 介词
    "在", "把", "被", "对", "从", "到", "向", "于",
    # 连词
    "和", "与", "及", "而", "但", "或", "则", "且",
    # 代词
    "我", "你", "他", "她", "它", "这", "那",
    # 副词
    "就", "都", "也", "还", "又", "再", "才", "只",
    "不", "没", "很", "最", "更", "非常",
    # 动词高频
    "是", "有", "要", "会", "能", "可以",
    # 量词/数词
    "个", "些", "点", "多", "少",
]


def extract_word_freq(text: str, top_n: int = 50) -> dict:
    """提取高频功能词的每千字频率"""
    chars = len(text)
    if chars == 0:
        return {}

    counts = {}
    for word in COMMON_FUNCTION_WORDS:
        n = text.count(word)
        counts[word] = round(n * 1000 / chars, 3)

    # 取 top_n
    sorted_words = sorted(counts.items(), key=lambda x: -x[1])
    return dict(sorted_words[:top_n])


def calc_delta(fp1: dict, fp2: dict, baseline_sd: dict = None) -> float:
    """计算 Burrows Delta（z-score 距离）"""
    common_words = set(fp1.keys()) & set(fp2.keys())
    if not common_words:
        return 999.0

    total = 0.0
    for word in common_words:
        v1 = fp1[word]
        v2 = fp2[word]
        if baseline_sd and word in baseline_sd:
            sd = baseline_sd[word]
            mean = 0  # 基线均值已归一化
        else:
            # 没有基线 sd 时，用两个文本的差值直接算
            sd = max(abs(v1 - v2), 1)
            mean = (v1 + v2) / 2

        if sd > 0:
            total += abs(v1 - v2) / sd

    return round(total / len(common_words), 3) if common_words else 999.0


def main():
    ap = argparse.ArgumentParser(description="Burrows Delta 作者识别")
    ap.add_argument("texts", nargs="*", help="对比的文本文件（2 个）")
    ap.add_argument("--corpus", help="语料目录（生成基线）")
    ap.add_argument("--baseline", help="基线 JSON（对比用）")
    ap.add_argument("--top", type=int, default=50, help="取 top N 高频词")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    if args.corpus:
        # 生成基线
        corpus_dir = Path(args.corpus)
        all_freqs = []
        for f in sorted(corpus.glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            freq = extract_word_freq(text, args.top)
            if freq:
                all_freqs.append(freq)

        # 算每词的均值和 sd
        baseline = {}
        all_words = set()
        for f in all_freqs:
            all_words.update(f.keys())

        for word in all_words:
            vals = [f[word] for f in all_freqs if word in f]
            if vals:
                mean = sum(vals) / len(vals)
                sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0
                baseline[word] = {"mean": round(mean, 3), "sd": round(sd, 3)}

        result = {
            "n_docs": len(all_freqs),
            "n_words": len(baseline),
            "baseline": baseline,
        }

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"基线生成完成：{len(all_freqs)} 篇 / {len(baseline)} 个高频词")
        return

    if len(args.texts) >= 2:
        text1 = Path(args.texts[0]).read_text(encoding="utf-8")
        text2 = Path(args.texts[1]).read_text(encoding="utf-8")
        fp1 = extract_word_freq(text1, args.top)
        fp2 = extract_word_freq(text2, args.top)

        baseline_sd = None
        if args.baseline:
            bl = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            baseline_sd = {w: v.get("sd", 0) for w, v in bl.get("baseline", {}).items()}

        delta = calc_delta(fp1, fp2, baseline_sd)

        if args.json:
            print(json.dumps({"burrows_delta": delta}, ensure_ascii=False))
        else:
            print(f"Burrows Delta: {delta}（越小越像）")
        return

    print("用法：burrows_delta.py <文本1> <文本2> 或 --corpus <目录>")


if __name__ == "__main__":
    main()
