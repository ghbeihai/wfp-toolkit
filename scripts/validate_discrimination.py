#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测判别力自检：验证基线尺子能不能分清"像不像"。

两步自检：
1. **留出集自检**：原作者语料留 20% 不进基线，看它 Delta 是不是明显好于随机对照文
2. **异作者判别**：拿另一个作者的文章跑，看 Delta 是不是明显差于原作者

两关过不了，Delta 就标"不可用"，不硬报数值。

用法：
  python3 validate_discrimination.py \
    --baseline <基线.json> \
    --held-out <原作者留出文章.txt> \
    --control <对照文.txt> \
    --other-author <另一作者文章.txt> [--json]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def extract_fp(text_file: str) -> dict:
    """调 extract_hard_metrics.py 提取指纹"""
    extract_script = Path(__file__).parent / "extract_hard_metrics.py"
    r = subprocess.run(
        [sys.executable, str(extract_script), text_file, "--json"],
        capture_output=True, text=True, encoding="utf-8"
    )
    if r.returncode != 0:
        raise RuntimeError(f"指纹提取失败: {r.stderr[:200]}")
    return json.loads(r.stdout)


def calc_delta(fp: dict, baseline: dict) -> float:
    """计算 z-score 距离（每千字归一化，排除篇幅指标）"""
    bl = baseline.get("baseline", baseline)
    chars = fp.get("chars", 0)

    # 排除篇幅指标
    exclude = {"chars", "sentences", "uniq_chars", "ttr_char"}
    # 按篇指标换算成每千字
    per_doc = {"L1_SYN_004_question_sents", "L1_SYN_004_rhet_sents"}
    # 字典型指标展开成总和
    dict_features = {"L1_LEX_004_yqci_per_k", "L1_LEX_006_pron_per_k"}

    total = 0.0
    n = 0

    for k, v in fp.items():
        if k in exclude:
            continue
        if isinstance(v, dict):
            if k in dict_features:
                v = sum(x for x in v.values() if isinstance(x, (int, float)))
            else:
                continue
        if not isinstance(v, (int, float)):
            continue
        if k in per_doc and chars > 0:
            v = v * 1000 / chars
        if k not in bl:
            continue
        bl_v = bl[k]
        if isinstance(bl_v, dict):
            mean = bl_v.get("mean", 0)
            sd = bl_v.get("sd", 0)
            # 基线按篇指标也需要换算成每千字
            if k in per_doc:
                avg_chars = baseline.get("avg_chars", 500)
                mean = mean * 1000 / avg_chars
                sd = sd * 1000 / avg_chars
        else:
            mean = bl_v
            sd = 0
        if sd > 0:
            total += abs(v - mean) / sd
            n += 1
    return round(total / n, 3) if n > 0 else 999.0


def main():
    ap = argparse.ArgumentParser(description="回测判别力自检")
    ap.add_argument("--baseline", required=True, help="基线 JSON")
    ap.add_argument("--held-out", required=True, help="原作者留出文章（不进基线的那 20%）")
    ap.add_argument("--control", required=True, help="对照文（随机文本或非模仿文）")
    ap.add_argument("--other-author", help="另一作者的文章（可选，测异作者判别）")
    ap.add_argument("--avg-chars", type=int, default=None, help="基线平均每篇字数（用于换算按篇指标）")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    # 提取三个指纹
    held_out_fp = extract_fp(args.held_out)
    control_fp = extract_fp(args.control)
    other_fp = extract_fp(args.other_author) if args.other_author else None

    # 算 Delta
    delta_held_out = calc_delta(held_out_fp, baseline)
    delta_control = calc_delta(control_fp, baseline)
    delta_other = calc_delta(other_fp, baseline) if other_fp else None

    # 判别比
    ratio_held_vs_control = round(delta_control / delta_held_out, 2) if delta_held_out > 0 else 0
    ratio_held_vs_other = round(delta_other / delta_held_out, 2) if (delta_held_out > 0 and delta_other) else None

    # 判断是否通过
    # 留出集应该比对照文好（判别比 > 1.5）
    passed_held_out = ratio_held_vs_control >= 1.5
    # 异作者应该比原作者差（判别比 > 1.2）
    passed_other = ratio_held_vs_other >= 1.2 if ratio_held_vs_other else None

    result = {
        "delta_held_out": delta_held_out,
        "delta_control": delta_control,
        "delta_other_author": delta_other,
        "ratio_held_vs_control": ratio_held_vs_control,
        "ratio_held_vs_other": ratio_held_vs_other,
        "passed_held_out_test": passed_held_out,
        "passed_other_author_test": passed_other,
        "verdict": "PASS" if (passed_held_out and (passed_other is None or passed_other)) else "FAIL",
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("=" * 50)
    print("回测判别力自检")
    print("=" * 50)
    print(f"原作者留出文 Delta: {delta_held_out}")
    print(f"对照文 Delta: {delta_control}")
    if delta_other:
        print(f"另一作者 Delta: {delta_other}")
    print()
    print(f"判别比（对照/原作者）: {ratio_held_vs_control} {'✓ 通过' if passed_held_out else '✗ 未通过'}")
    if ratio_held_vs_other:
        print(f"判别比（另一作者/原作者）: {ratio_held_vs_other} {'✓ 通过' if passed_other else '✗ 未通过'}")
    print()
    if result["verdict"] == "PASS":
        print("结论：尺子可用，Delta 结果可信。")
    else:
        print("结论：尺子不可用，Delta 结果标注为'不可信'。")
        print("可能原因：语料太短、异质语料、基线 sd 太大。")
        print("建议：用 ≥500 字的块、≥20 篇语料重跑基线。")


if __name__ == "__main__":
    main()
