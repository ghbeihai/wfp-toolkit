#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测判别力自检：验证基线尺子能不能分清"像不像"。

支持两种模式：
1. 手动模式：指定 --held-out 和 --control 跑一对
2. 自动模式：指定 --corpus 语料目录，自动留出 20%，跑多组，汇总通过率

用法：
  # 手动模式
  python3 validate_discrimination.py --baseline baseline.json --held-out held.txt --control control.txt

  # 自动模式（自动留出 20%）
  python3 validate_discrimination.py --corpus ./原文目录 --baseline baseline.json --holdout 20
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


EXCLUDE = {
    "chars", "sentences", "uniq_chars",
    "L1_LEX_001_ttr_char",
    "L2_DIS_001_avg_para_len",
}

PER_DOC = {
    "L1_SYN_004_question_sents",
    "L1_SYN_004_rhet_sents",
}

DICT_FEATURES = {
    "L1_LEX_004_yqci_per_k",
    "L1_LEX_006_pron_per_k",
    "L1_LEX_005_dm_per_k",
}

# 展开后的标量字段（基线聚合时生成的 *_total）
DICT_TOTAL_FEATURES = {
    "L1_LEX_004_yqci_per_k_total",
    "L1_LEX_006_pron_per_k_total",
    "L1_LEX_005_dm_total",
}


def extract_fp(text_file: str) -> dict:
    extract_script = Path(__file__).parent / "extract_hard_metrics.py"
    r = subprocess.run(
        [sys.executable, str(extract_script), text_file, "--json"],
        capture_output=True, text=True, encoding="utf-8"
    )
    if r.returncode != 0:
        raise RuntimeError(f"指纹提取失败: {r.stderr[:200]}")
    return json.loads(r.stdout)


def calc_delta(fp: dict, baseline: dict, avg_chars_override: int = None) -> float:
    bl = baseline.get("baseline", baseline)
    chars = fp.get("chars", 0)
    avg_chars = avg_chars_override or baseline.get("avg_chars", 500)

    total = 0.0
    n = 0

    for k, v in fp.items():
        if k in EXCLUDE:
            continue
        if isinstance(v, dict):
            if k in DICT_FEATURES:
                v = sum(x for x in v.values() if isinstance(x, (int, float)))
            else:
                continue
        if not isinstance(v, (int, float)):
            continue
        if k in PER_DOC and chars > 0:
            v = v * 1000 / chars
        if k not in bl:
            continue
        bl_v = bl[k]
        if isinstance(bl_v, dict):
            mean = bl_v.get("mean", 0)
            sd = bl_v.get("sd", 0)
            if k in PER_DOC:
                mean = mean * 1000 / avg_chars
                sd = sd * 1000 / avg_chars
        else:
            mean = bl_v
            sd = 0
        if sd > 0:
            total += abs(v - mean) / sd
            n += 1
    return round(total / n, 3) if n > 0 else 999.0


def run_one_test(held_out_file: str, control_file: str, baseline: dict, avg_chars: int = None) -> dict:
    held_out_fp = extract_fp(held_out_file)
    control_fp = extract_fp(control_file)
    delta_held = calc_delta(held_out_fp, baseline, avg_chars)
    delta_control = calc_delta(control_fp, baseline, avg_chars)
    ratio = round(delta_control / delta_held, 2) if delta_held > 0 else 0
    passed = ratio >= 1.5
    return {
        "delta_held": delta_held,
        "delta_control": delta_control,
        "ratio": ratio,
        "passed": passed,
    }


def main():
    ap = argparse.ArgumentParser(
        description="回测判别力自检"
    )
    ap.add_argument("--baseline", required=True, help="基线 JSON")
    ap.add_argument("--held-out", help="留出文章（手动模式）")
    ap.add_argument("--control", help="对照文（手动模式）")
    ap.add_argument("--corpus", help="语料目录（自动模式）")
    ap.add_argument("--other-author", help="另一作者的语料目录（自动模式的对照组）")
    ap.add_argument("--holdout", type=int, default=20, help="自动留出比例（默认 20%%）")
    ap.add_argument("--rounds", type=int, default=5, help="自动模式跑几轮（默认 5 轮）")
    ap.add_argument("--avg-chars", type=int, default=None, help="基线平均每篇字数")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    # 手动模式
    if args.held_out and args.control:
        result = run_one_test(args.held_out, args.control, baseline, args.avg_chars)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"原作者留出 Delta: {result['delta_held']}")
            print(f"对照文 Delta: {result['delta_control']}")
            print(f"判别比: {result['ratio']} {'PASS' if result['passed'] else 'FAIL'}")
        return

    # 自动模式
    if args.corpus:
        corpus_dir = Path(args.corpus)
        files = sorted(corpus_dir.glob("*.txt"))
        if len(files) < 10:
            print("语料太少，至少 10 篇才能自动留出测试。", file=sys.stderr)
            sys.exit(1)

        results = []
        for r in range(args.rounds):
            # 随机留出
            n_holdout = max(1, int(len(files) * args.holdout / 100))
            random.shuffle(files)
            holdout = files[:n_holdout]
            if args.other_author:
                other_files = sorted(Path(args.other_author).glob("*.txt"))
                control_files = other_files[:n_holdout]
                control_text = "\n".join(f.read_text(encoding="utf-8") for f in control_files)
            else:
                control = files[n_holdout:n_holdout + n_holdout]  # 用剩下的当对照（警告：同作者对比意义有限）
                control_text = "\n".join(f.read_text(encoding="utf-8") for f in control)

            # 拼起来当留出文
            held_text = "\n".join(f.read_text(encoding="utf-8") for f in holdout)

            held_file = "/tmp/_held.txt"
            control_file = "/tmp/_control.txt"
            Path(held_file).write_text(held_text, encoding="utf-8")
            Path(control_file).write_text(control_text, encoding="utf-8")

            res = run_one_test(held_file, control_file, baseline, args.avg_chars)
            results.append(res)
            print(f"第 {r+1} 轮: 判别比 {res['ratio']} {'PASS' if res['passed'] else 'FAIL'}")

        # 汇总通过率
        n_pass = sum(1 for r in results if r["passed"])
        pass_rate = round(n_pass / len(results) * 100, 1)
        avg_ratio = round(sum(r["ratio"] for r in results) / len(results), 2)

        print(f"\n汇总：{len(results)} 轮，通过 {n_pass} 轮，通过率 {pass_rate}%")
        print(f"平均判别比: {avg_ratio}")
        print(f"结论: {'PASS' if pass_rate >= 60 else 'FAIL'}（通过率 ≥60% 才算尺子可用）")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
