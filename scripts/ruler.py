#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多尺子（multi-ruler）：WFP 分层子尺 L1 句法 / L1 词汇 / L2 语篇 / L3 软评分 / L4 隐喻与价值（含 LLM 隐喻），
加权合成综合分；权重由自检判别力学出（原作者留出 vs 对照/异作者的可分性），
不可判别或与篇长强相关的特征自动降权（只降权、不删除——任何层都保留在尺子里）。

特征来源：
  - 硬指标：field_map.flatten(指纹) 的 Delta 特征（L1/L2/L4 规则版）；
  - L4_LLM_meta_per_k：score_metaphor.py 的 LLM 隐喻密度（指纹里有该字段、或用 --metaphor-dir 附上）；
  - L3::<维度>：score_stylistics.py 的 17 维软评分（--l3-dir），基线来自 l3_baseline.json。
基线扩展：ruler.py extend 把 LLM 隐喻 / L3 的 mean/sd 写进基线的 "extra" 节。

用法：
  python3 ruler.py extend --baseline bl.json [--metaphor-dir M] [--l3-baseline l3bl.json] --out bl_ext.json
  python3 ruler.py learn  --baseline bl_ext.json --pos 留出指纹目录 --neg 对照指纹目录 [--neg 他人指纹目录] \\
                          [--length-fps 基线语料指纹目录] [--metaphor-dir M] [--l3-dir L] --out weights.json
  python3 ruler.py score  指纹.json --baseline bl_ext.json [--weights weights.json] [--metaphor-dir M] [--l3-dir L]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import field_map as fm

LAYERS = {"L1_SYN": "L1 句法", "L1_LEX": "L1 词汇", "L2": "L2 语篇", "L3": "L3 软评分", "L4": "L4 隐喻/价值"}
META_KEY = "L4_LLM_meta_per_k"
LEN_RHO = 0.4        # |spearman(特征, 篇长)| ≥ 此值视为篇长相关
LEN_FACTOR = 0.5     # 篇长相关特征权重 × 此系数
FLOOR = 0.05         # 不可判别特征保留的最低权重比例（不删除）
WEIGHTS_VERSION = 1


def layer_of(key: str) -> str:
    if key.startswith("L3::") or key.startswith("L3_"):
        return "L3"
    for p in ("L1_SYN", "L1_LEX", "L2", "L4"):
        if key.startswith(p):
            return p
    return "other"


def prior(key: str) -> float:
    return fm.weight(key) if not key.startswith("L3::") and key != META_KEY else 1.0


# ---------- 特征向量 ----------
def load_metaphor(mdir, doc_id):
    if not mdir:
        return None
    p = Path(mdir) / f"{doc_id}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("failed_chunks"):
            return d.get("meta_per_k")
    return None


def load_l3_dir(l3dir) -> dict:
    """score_stylistics 输出目录 → {doc_id: {dim: score}}（跳过失败/模拟）。"""
    out = {}
    if not l3dir:
        return out
    for f in Path(l3dir).glob("*.json"):
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            if not isinstance(v, dict) or "::" not in k:
                continue
            s = v.get("score")
            if isinstance(s, (int, float)) and s > 0 and not v.get("failed") and not v.get("simulated"):
                doc, dim = k.rsplit("::", 1)
                out.setdefault(doc, {})[dim] = s
    return out


def vector(fp: dict, metaphor_dir=None, l3=None) -> dict:
    v = fm.flatten(fp)
    m = fp.get(META_KEY)
    if m is None:
        m = load_metaphor(metaphor_dir, fp.get("doc_id", ""))
    if isinstance(m, (int, float)):
        v[META_KEY] = m
    for dim, s in ((l3 or {}).get(str(fp.get("doc_id", "")), {}) or {}).items():
        v[f"L3::{dim}"] = s
    return v


def stat(baseline: dict, key: str):
    ex = baseline.get("extra", {}).get(key)
    if ex:
        return ex["mean"], ex["sd"]
    if key.startswith("L3::") or key == META_KEY:
        return None
    return fm.baseline_stat(baseline, key)


def zscores(vec: dict, baseline: dict) -> dict:
    zs = {}
    for k, x in vec.items():
        st = stat(baseline, k)
        if st and st[1]:
            zs[k] = abs(x - st[0]) / st[1]
    return zs


# ---------- 打分 ----------
def score(vec: dict, baseline: dict, weights: dict | None = None) -> dict:
    """返回 {composite, n, layers:{层:{score, n, weight_share}}, per_feature:{k: z}}。越低越像。"""
    fm.check_baseline(baseline)
    zs = zscores(vec, baseline)
    W = (weights or {}).get("features", {})
    w = {k: (W[k]["weight"] if k in W else prior(k)) for k in zs}
    tot = sum(w.values())
    layers = {}
    for k, z in zs.items():
        L = layers.setdefault(layer_of(k), {"num": 0.0, "w": 0.0, "n": 0, "zs": []})
        L["num"] += z * w[k]
        L["w"] += w[k]
        L["n"] += 1
        L["zs"].append(z)
    out_layers = {}
    for name, L in sorted(layers.items()):
        out_layers[name] = {"label": LAYERS.get(name, name),
                            "score": round(L["num"] / L["w"], 4) if L["w"] else round(statistics.mean(L["zs"]), 4),
                            "n": L["n"], "weight_share": round(L["w"] / tot, 4) if tot else 0.0}
    comp = sum(z * w[k] for k, z in zs.items()) / tot if tot else None
    return {"composite": None if comp is None else round(comp, 4), "n": len(zs), "layers": out_layers,
            "per_feature": {k: round(z, 3) for k, z in sorted(zs.items(), key=lambda kv: -kv[1])}}


# ---------- 学权重 ----------
def _auc(pos, neg):
    """P(z_neg > z_pos)（Mann-Whitney，平局记 0.5）。"""
    if not pos or not neg:
        return 0.5
    s = 0.0
    for a in neg:
        for b in pos:
            s += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s / (len(pos) * len(neg))


def _spearman(x, y):
    n = len(x)
    if n < 5:
        return 0.0

    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def learn(pos_vecs: list, neg_vecs: list, baseline: dict, length_rows: list | None = None) -> dict:
    """pos = 原作者（应当近），neg = 对照/异作者（应当远）。返回 weights JSON。"""
    zp = [zscores(v, baseline) for v in pos_vecs]
    zn = [zscores(v, baseline) for v in neg_vecs]
    keys = sorted({k for z in zp + zn for k in z})
    feats = {}
    for k in keys:
        p = [z[k] for z in zp if k in z]
        q = [z[k] for z in zn if k in z]
        auc = _auc(p, q)
        disc = max(0.0, 2 * auc - 1)
        reasons = []
        w = prior(k) * max(disc, FLOOR)
        if disc < 0.1:
            reasons.append(f"判别力弱（AUC={auc:.2f}）→ 降到 {FLOOR:.0%} 下限附近")
        rho = 0.0
        if length_rows:
            pairs = [(r[0][k], r[1]) for r in length_rows if k in r[0]]
            if len(pairs) >= 5:
                rho = _spearman([a for a, _ in pairs], [b for _, b in pairs])
                if abs(rho) >= LEN_RHO:
                    w *= LEN_FACTOR
                    reasons.append(f"与篇长相关（ρ={rho:+.2f}）→ ×{LEN_FACTOR}")
        feats[k] = {"layer": layer_of(k), "prior": round(prior(k), 4), "auc": round(auc, 3),
                    "disc": round(disc, 3), "length_rho": round(rho, 3), "raw": w, "reasons": reasons,
                    "n_pos": len(p), "n_neg": len(q)}
    # 归一：总权重 = 先验总和（让综合分与旧 Delta 同量级）
    s_raw = sum(f["raw"] for f in feats.values()) or 1.0
    s_pri = sum(f["prior"] for f in feats.values())
    for f in feats.values():
        f["weight"] = round(f.pop("raw") * s_pri / s_raw, 5)
    tot = sum(f["weight"] for f in feats.values()) or 1.0
    layers = {}
    for f in feats.values():
        L = layers.setdefault(f["layer"], {"label": LAYERS.get(f["layer"], f["layer"]), "weight_share": 0.0, "n": 0})
        L["weight_share"] += f["weight"] / tot
        L["n"] += 1
    for L in layers.values():
        L["weight_share"] = round(L["weight_share"], 4)
    return {"version": WEIGHTS_VERSION, "field_map_version": fm.FIELD_MAP_VERSION,
            "n_pos": len(pos_vecs), "n_neg": len(neg_vecs), "layers": layers, "features": feats}


# ---------- 基线扩展 ----------
def extend_baseline(baseline: dict, metaphor_dir=None, l3_baseline: dict | None = None) -> dict:
    out = dict(baseline)
    extra = dict(baseline.get("extra", {}))
    if metaphor_dir:
        vals = []
        for d in baseline.get("doc_ids", []):
            m = load_metaphor(metaphor_dir, d)
            if m is not None:
                vals.append(m)
        if len(vals) >= 5:
            extra[META_KEY] = {"mean": round(statistics.mean(vals), 4), "sd": round(statistics.stdev(vals), 4),
                               "n": len(vals), "source": "score_metaphor"}
        else:
            print(f"[warn] LLM 隐喻只有 {len(vals)} 篇基线文档有结果（<5），未加入基线", file=sys.stderr)
    for dim, v in ((l3_baseline or {}).get("baseline") or {}).items():
        if v.get("sd"):
            extra[f"L3::{dim}"] = {"mean": v["mean"], "sd": v["sd"], "n": v.get("n"), "source": "l3_baseline",
                                   "dim_name": v.get("dim_name")}
    out["extra"] = extra
    return out


def _load_fps(d):
    p = Path(d)
    files = [p] if p.is_file() else sorted(p.glob("*.json"))
    out = []
    for f in files:
        fp = json.loads(f.read_text(encoding="utf-8"))
        fp.setdefault("doc_id", f.stem)
        out.append(fp)
    return out


def main():
    ap = argparse.ArgumentParser(description="WFP 多尺子：分层子尺 + 学习权重的综合分")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extend", help="把 LLM 隐喻 / L3 基线并入 WFP 基线（extra 节）")
    e.add_argument("--baseline", required=True)
    e.add_argument("--metaphor-dir")
    e.add_argument("--l3-baseline")
    e.add_argument("--out", required=True)
    le = sub.add_parser("learn", help="由自检判别力学习特征/层权重")
    le.add_argument("--baseline", required=True)
    le.add_argument("--pos", action="append", required=True, help="原作者留出指纹目录（可多次）")
    le.add_argument("--neg", action="append", required=True, help="对照/异作者指纹目录（可多次）")
    le.add_argument("--length-fps", help="基线语料指纹目录（检测与篇长相关的特征）")
    le.add_argument("--metaphor-dir")
    le.add_argument("--l3-dir")
    le.add_argument("--out", required=True)
    sc = sub.add_parser("score", help="对指纹打综合分 + 分层分")
    sc.add_argument("fps", nargs="+")
    sc.add_argument("--baseline", required=True)
    sc.add_argument("--weights")
    sc.add_argument("--metaphor-dir")
    sc.add_argument("--l3-dir")
    sc.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        bl = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        if a.cmd == "extend":
            l3 = json.loads(Path(a.l3_baseline).read_text(encoding="utf-8")) if a.l3_baseline else None
            out = extend_baseline(bl, a.metaphor_dir, l3)
            Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] {a.out}：extra 特征 {sorted(out['extra'])}")
            return
        l3 = load_l3_dir(a.l3_dir)
        if a.cmd == "learn":
            fm.check_baseline(bl)
            pos = [vector(fp, a.metaphor_dir, l3) for d in a.pos for fp in _load_fps(d)]
            neg = [vector(fp, a.metaphor_dir, l3) for d in a.neg for fp in _load_fps(d)]
            lr = None
            if a.length_fps:
                lr = [(vector(fp, a.metaphor_dir, l3), fp["chars"]) for fp in _load_fps(a.length_fps)]
            w = learn(pos, neg, bl, lr)
            Path(a.out).write_text(json.dumps(w, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] 权重 {a.out}（正 {w['n_pos']} / 负 {w['n_neg']}）")
            for k, L in sorted(w["layers"].items()):
                print(f"  {L['label']}: 权重占比 {L['weight_share']:.1%}（{L['n']} 项）")
            return
        W = json.loads(Path(a.weights).read_text(encoding="utf-8")) if a.weights else None
        res = []
        for f in a.fps:
            for fp in _load_fps(f):
                r = score(vector(fp, a.metaphor_dir, l3), bl, W)
                r["doc_id"] = fp.get("doc_id")
                res.append(r)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return
        for r in res:
            ls = "；".join(f"{L['label']} {L['score']}" for L in r["layers"].values())
            print(f"{r['doc_id']}: 综合 {r['composite']}（{r['n']} 项）| {ls}")
    except (fm.DegenerateBaseline, fm.EmptyText) as e:
        sys.exit(f"错误：{e}")


if __name__ == "__main__":
    main()
