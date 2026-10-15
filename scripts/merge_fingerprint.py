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
import statistics

# 参与 Delta 计算的标量特征（全部来自已核验的 L1/L2/L4 硬指标）
SCALAR_FEATURES = [
    "L1_LEX_001_ttr_char", "L1_LEX_001_sttr100", "L1_LEX_002_ttr_word",
    "L1_LEX_006_wo_ni_ratio", "L1_LEX_007_pos_per_k", "L1_LEX_007_neg_per_k",
    "L1_SYN_001_avg_sl", "L1_SYN_002_sd_sl", "L1_SYN_002_med_sl",
    "L1_SYN_002_short_lt10", "L1_SYN_002_long_gt40",
    "L1_SYN_004_question_sents", "L1_SYN_004_rhet_sents",
    "L1_SYN_005_ba_per_k", "L1_SYN_005_bei_per_k", "L1_SYN_005_rang_per_k",
    "L1_SYN_005_shi_de_per_k", "L1_SYN_006_fourchar_per_k",
    "L1_SYN_007_burstiness",
    "L2_DIS_001_paras", "L2_DIS_001_avg_para_len",
    "L2_DIS_003_pron_per_sent",
    "L4_SUR_001_meta_per_k", "L4_SUR_002_value_per_k",
    "L4_SUR_003_pos_per_k", "L4_SUR_003_neg_per_k", "L4_SUR_003_pos_neg_ratio",
    # 派生标量（v0.4.1：嵌套字段聚合，纳入 Delta 池）
    "L1_LEX_004_yqci_total_per_k", "L2_DIS_002_conn_total_per_k",
    "L4_SUR_001_meta_top1_pct",
]


def derive(fp: dict) -> dict:
    """从嵌套字段派生标量（操作化口径）：语气词/衔接词总和、根域 Top1 占比。"""
    out = {}
    yq = fp.get("L1_LEX_004_yqci_per_k", {})
    out["L1_LEX_004_yqci_total_per_k"] = round(sum(yq.values()), 3) if yq else 0.0
    conn = fp.get("L2_DIS_002_conn_per_k", {})
    out["L2_DIS_002_conn_total_per_k"] = round(sum(conn.values()), 3) if conn else 0.0
    md = fp.get("L4_SUR_001_meta_domains", {})
    if md and sum(md.values()) > 0:
        top = max(md.values())
        out["L4_SUR_001_meta_top1_pct"] = round(top / sum(md.values()), 4)
    else:
        out["L4_SUR_001_meta_top1_pct"] = 0.0
    return out


def load_baseline(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    return d["baseline"]


def vector(fp: dict) -> dict:
    v = dict(fp)
    v.update(derive(fp))
    return {k: v.get(k) for k in SCALAR_FEATURES if isinstance(v.get(k), (int, float))}


def load_l3(doc_id: str, l3dir: str | None) -> dict | None:
    """从 L3 评分目录读取该文档的软评分（如有），返回 {dim: score}。"""
    if not l3dir:
        return None
    import glob
    cands = glob.glob(f"{l3dir}/*.json")
    for f in cands:
        d = json.load(open(f, encoding="utf-8"))
        if any(k.split("::")[0] == doc_id for k in d):
            return {k.split("::")[1]: v["score"] for k, v in d.items() if v.get("score", 0) > 0}
    return None


def delta(fp: dict, baseline: dict) -> dict:
    """Burrows 2002 Delta：z-score 化后 |z| 均值。返回逐特征 z 与总 Delta。
    v0.4.19.1：burstiness 按样本句数自动选档（15-45 句→short 基线，≥50 句→long 基线）。"""
    zs = {}
    n_sent = fp.get("sentences", 0)
    for k, x in vector(fp).items():
        b = baseline.get(k)
        if k == "L1_SYN_007_burstiness" and b:
            if 15 <= n_sent <= 45:
                b = baseline.get("L1_SYN_007_burstiness__short") or b
            elif n_sent >= 50:
                b = baseline.get("L1_SYN_007_burstiness__long") or b
        if not b or b.get("sd", 0) == 0 or x is None:
            continue
        zs[k] = round((x - b["mean"]) / b["sd"], 3)
    if not zs:
        return {"delta": None, "n": 0, "zs": {}}
    return {"delta": round(sum(abs(v) for v in zs.values()) / len(zs), 4),
            "n": len(zs), "zs": zs}


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
    out = []
    for f in args.files:
        fp = json.load(open(f, encoding="utf-8"))
        d = delta(fp, baseline)
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
