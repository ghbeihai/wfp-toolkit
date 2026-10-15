#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WFP · 文风指纹 — 指纹合并与 Delta 距离（v0.4）

- 合并：把单篇指纹 JSON（L1/L2/L4 硬指标）整理为标量特征向量 + 可读摘要
- 距离：Burrows 2002「Delta」口径——特征 z-score 化后与基线比较
  （z = (x - mean) / sd；Delta = 特征级 |z| 均值；越小越接近基线）
- 回测：真实文 / 模仿文 / 对照文 与目标作者基线的距离比较

用法：
    python3 scripts/merge_fingerprint.py <指纹JSON> --baseline <基线JSON> --json
    python3 scripts/merge_fingerprint.py <指纹JSON1> <指纹JSON2> ... --baseline <基线JSON>   # 对比
"""
import argparse
import json

# 参与 Delta 计算的特征与口径：统一从 field_map 读，禁止各自维护
import sys

from field_map import delta_features, flatten, delta as fm_delta, check_baseline, DegenerateBaseline, EmptyText
SCALAR_FEATURES = delta_features()


def load_baseline(path: str) -> dict:
    """返回完整基线 JSON（含 avg_chars，供旧基线口径兼容换算）。"""
    d = json.load(open(path, encoding="utf-8"))
    return d if "baseline" in d else {"baseline": d}


def vector(fp: dict) -> dict:
    """Delta 特征向量（与 normalize_backtest / validate_discrimination 同一实现）。"""
    return flatten(fp)


def load_l3(doc_id: str, l3dir: str | None) -> dict | None:
    """从 L3 评分目录读取该文档的软评分（如有），返回 {dim: score}。"""
    if not l3dir:
        return None
    import glob
    cands = glob.glob(f"{l3dir}/*.json")
    for f in cands:
        d = json.load(open(f, encoding="utf-8"))
        if any(k.split("::")[0] == doc_id for k in d):
            return {k.split("::")[1]: v["score"] for k, v in d.items()
                    if isinstance(v.get("score"), (int, float)) and v["score"] > 0}
    return None


def delta(fp: dict, baseline: dict) -> dict:
    """Burrows 2002 Delta：z-score 化后 |z| 均值。返回逐特征 z 与总 Delta。
    口径由 field_map.delta 统一（v1.4.4 起不再查不存在的 burstiness__short/long）。"""
    r = fm_delta(fp, baseline if "baseline" in baseline else {"baseline": baseline},
                 return_detail=True)
    return {"delta": r["delta"], "n": r["n"], "zs": r["zs"]}


def summary(fp: dict) -> dict:
    return {
        "doc": fp.get("doc_id", "")[:40],
        "chars": fp.get("chars"),
        "sentences": fp.get("sentences"),
        "paras": fp.get("L2_DIS_001_paras"),
        "avg_sl": fp.get("L1_SYN_001_avg_sl"),
        "sttr": fp.get("L1_LEX_001_sttr100"),
        "wo_ni": fp.get("L1_LEX_006_wo_ni_ratio"),
        "questions": fp.get("L1_SYN_004_question_sents"),
        "meta_per_k": fp.get("L4_SUR_001_meta_per_k"),
        "top_domain": max(fp.get("L4_SUR_001_meta_domains", {}).items(), key=lambda x: x[1])[0]
        if fp.get("L4_SUR_001_meta_domains") else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="指纹 JSON 路径（1 个或多个）")
    ap.add_argument("--baseline", required=True, help="基线 JSON（wfp_baseline_*.json）")
    ap.add_argument("--l3dir", default=None, help="L3 评分目录（data/results/l3_scores），融合软评分")
    ap.add_argument("--l3-baseline", default=None, help="L3 基线 JSON（计算软画像偏差）")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = ap.parse_args()

    baseline = load_baseline(args.baseline)
    try:
        check_baseline(baseline)
    except DegenerateBaseline as e:
        sys.exit(f"错误：{e}")
    out = []
    for f in args.files:
        fp = json.load(open(f, encoding="utf-8"))
        try:
            d = delta(fp, baseline)
        except EmptyText as e:
            sys.exit(f"错误：{f}: {e}")
        item = {"file": f, "summary": summary(fp), "delta": d["delta"],
                "n_features": d["n"], "zs": d["zs"]}
        l3 = load_l3(fp.get("doc_id", ""), args.l3dir)

        # L3 偏差计算
        l3_deviation = None
        if l3 and args.l3_baseline:
            l3_bl = json.loads(open(args.l3_baseline, encoding="utf-8").read())
            l3_bl_mean = l3_bl.get("baseline", l3_bl)
            deviations = []
            for k, v in l3.items():
                if k in l3_bl_mean:
                    bl_v = l3_bl_mean[k]
                    if isinstance(bl_v, dict):
                        mean = bl_v.get("mean", 0)
                        sd = bl_v.get("sd", 1)
                        if sd > 0:
                            deviations.append(abs(v - mean) / sd)
            l3_deviation = round(sum(deviations) / len(deviations), 3) if deviations else None
        if l3_deviation is not None:
            item["l3_deviation"] = l3_deviation
        if l3:
            item["l3_scores"] = l3
            item["l3_mean"] = round(sum(l3.values()) / len(l3), 2)
        out.append(item)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"基线: {args.baseline}")
    for o in sorted(out, key=lambda x: x["delta"] or 9e9):
        s = o["summary"]
        extra = f" | L3均值{o['l3_mean']}" if o.get("l3_mean") else ""
        print(f"  Delta={o['delta']} (n={o['n_features']}) | {s['doc']}{extra}")
        print(f"     句长{s['avg_sl']} STTR{s['sttr']} 问句{s['questions']} "
              f"我:你{s['wo_ni']} 隐喻{s['meta_per_k']}/千字 根域[{s['top_domain']}]")


if __name__ == "__main__":
    main()
