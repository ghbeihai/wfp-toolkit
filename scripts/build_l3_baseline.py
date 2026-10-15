#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 软评分基线生成：从多篇 L3 评分结果汇总成基线。

用法：
  python3 build_l3_baseline.py <l3_scores_dir> --out l3_baseline.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="L3 软评分基线生成")
    ap.add_argument("scores_dir", help="L3 评分结果目录（多篇 JSON）")
    ap.add_argument("--out", default="l3_baseline.json", help="输出文件名")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    if not scores_dir.exists():
        print(f"错误：目录不存在 {scores_dir}", file=sys.stderr)
        sys.exit(1)

    # 按维度汇总所有篇目的分数
    dim_scores = {}
    n_docs = 0

    for f in sorted(scores_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 跳过 {f.name}: {e}", file=sys.stderr)
            continue
        n_docs += 1

        # 遍历所有维度
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            dim = entry.get("dim", "")
            score = entry.get("score")
            dim_name = entry.get("dim_name", dim)
            if (not isinstance(score, (int, float)) or score <= 0
                    or entry.get("failed") or entry.get("simulated")):
                continue  # 失败/模拟分不进基线
            if dim not in dim_scores:
                dim_scores[dim] = {"scores": [], "dim_name": dim_name}
            dim_scores[dim]["scores"].append(score)

    if not dim_scores:
        print(f"错误：{scores_dir} 下没有可用的 L3 评分（0 篇有效 JSON / 0 个有效维度），未写出基线。",
              file=sys.stderr)
        sys.exit(1)

    # 算每维均值和 sd
    baseline = {}
    for dim, data in dim_scores.items():
        scores = data["scores"]
        baseline[dim] = {
            "dim_name": data["dim_name"],
            "mean": round(statistics.mean(scores), 3),
            "sd": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
            "n": len(scores),
        }

    result = {
        "n_docs": n_docs,
        "n_dims": len(baseline),
        "baseline": baseline,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"L3 基线生成完成：{n_docs} 篇 → {len(baseline)} 维 → {out_path}")


if __name__ == "__main__":
    main()
