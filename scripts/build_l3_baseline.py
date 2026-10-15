#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 软评分基线生成：从多篇 L3 评分结果汇总成基线。
跳过 failed / simulated / 非数值分；sd 用样本标准差（n-1）；无有效评分时报错退出。

用法：
  python3 build_l3_baseline.py <l3_scores_dir> --out l3_baseline.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


class NoValidScores(ValueError):
    pass


def build_baseline(scores_dir) -> dict:
    """汇总目录下全部 L3 评分 JSON → {n_docs, n_dims, n_failed, baseline{dim: {dim_name, mean, sd, n}}}。"""
    scores_dir = Path(scores_dir)
    if not scores_dir.exists():
        raise NoValidScores(f"目录不存在 {scores_dir}")
    dim_scores = {}
    docs = set()
    n_failed = 0
    for f in sorted(scores_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] 跳过 {f.name}: {e}", file=sys.stderr)
            continue
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            dim = entry.get("dim") or key.rsplit("::", 1)[-1]
            score = entry.get("score")
            if (not isinstance(score, (int, float)) or score <= 0
                    or entry.get("failed") or entry.get("simulated")):
                n_failed += 1
                continue  # 失败/模拟分不进基线
            docs.add(key.rsplit("::", 1)[0] if "::" in key else f.stem)
            d = dim_scores.setdefault(dim, {"scores": [], "dim_name": entry.get("dim_name", dim)})
            d["scores"].append(score)
    if not dim_scores:
        raise NoValidScores(f"{scores_dir} 下没有可用的 L3 评分（0 个有效维度），未写出基线")
    baseline = {}
    for dim, d in dim_scores.items():
        s = d["scores"]
        baseline[dim] = {"dim_name": d["dim_name"], "mean": round(statistics.mean(s), 3),
                         "sd": round(statistics.stdev(s), 3) if len(s) > 1 else 0.0, "n": len(s)}
    if n_failed:
        print(f"[warn] 跳过 {n_failed} 条失败/模拟/无效评分", file=sys.stderr)
    return {"n_docs": len(docs), "n_dims": len(baseline), "n_skipped": n_failed, "baseline": baseline}


def main():
    ap = argparse.ArgumentParser(description="L3 软评分基线生成")
    ap.add_argument("scores_dir", help="L3 评分结果目录（多篇 JSON）")
    ap.add_argument("--out", default="l3_baseline.json", help="输出文件名")
    args = ap.parse_args()
    try:
        result = build_baseline(args.scores_dir)
    except NoValidScores as e:
        sys.exit(f"错误：{e}")
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"L3 基线生成完成：{result['n_docs']} 篇 → {result['n_dims']} 维 → {out_path}")


if __name__ == "__main__":
    main()
