#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一字段映射表：指纹键 ↔ 基线 *_total。

所有脚本（validate_discrimination / normalize_backtest / merge_fingerprint）
都从这里读字段名，禁止各自维护一份。
"""

# ============ 标量字段（指纹和基线同名，直接对比）============
SCALAR_FIELDS = {
    # 句法层
    "L1_SYN_001_avg_sl": "平均句长",
    "L1_SYN_002_sd_sl": "句长标准差",
    "L1_SYN_002_short_lt10": "短句<10字占比",
    "L1_SYN_002_long_gt40": "长句>40字占比",
    "L1_SYN_002_med_sl": "中位句长",
    "L1_SYN_007_burstiness": "burstiness",
    "L1_SYN_005_ba_per_k": "把字句/千字",
    "L1_SYN_005_bei_per_k": "被字句/千字",
    "L1_SYN_005_rang_per_k": "让字句/千字",
    "L1_SYN_005_shi_de_per_k": "是...的/千字",
    # 词汇层
    "L1_LEX_001_sttr100": "STTR100",
    "L1_LEX_007_pos_per_k": "正情/千字",
    "L1_LEX_007_neg_per_k": "负情/千字",
    "L1_SYN_006_fourchar_per_k": "四字格/千字",
    "L1_LEX_006_wo_ni_ratio": "我:你比例",
    # 语篇层
    "L2_DIS_003_pron_per_sent": "每句代词数",
    # L4 深层
    "L4_SUR_001_meta_per_k": "隐喻/千字",
    "L4_SUR_002_value_per_k": "价值词/千字",
    "L4_SUR_003_pos_neg_ratio": "正负情感比",
}

# ============ 按篇指标（需要按 avg_chars 换算成每千字）============
# 键 = 指纹里的按篇计数，值 = 换算后的每千字字段名（Delta / 基线统一用后者）
PER_DOC_FIELDS = {
    "L1_SYN_004_question_sents": "L1_SYN_004_question_per_k",   # 问句
    "L1_SYN_004_rhet_sents": "L1_SYN_004_rhet_per_k",           # 修辞反问
}

# ============ 字典字段（指纹是字典，基线是 *_total 标量）============
# 键 = 指纹里的原始字段名，值 = 基线里的 *_total 字段名
DICT_TO_TOTAL = {
    "L1_LEX_004_yqci_per_k": "L1_LEX_004_yqci_per_k_total",      # 语气词
    "L1_LEX_006_pron_per_k": "L1_LEX_006_pron_per_k_total",      # 人称代词
    "L1_LEX_005_dm": "L1_LEX_005_dm_per_k_total",               # 话语标记（原始计数，按 chars 换算）
    "L2_DIS_002_conn_per_k": "L2_DIS_002_conn_per_k_total",     # 连接词
    "L1_LEX_008_cog_per_k": "L1_LEX_008_cog_per_k_total",       # 认知词
    "L1_SYN_003_punct_per_k": "L1_SYN_003_punct_per_k_total",     # 标点
}

# 字典值是"按篇原始计数"而非每千字的字段：求和后需 ×1000/chars
COUNT_DICTS = {"L1_LEX_005_dm"}

# ============ 排除字段（篇幅量 / 重复，不参与 Delta）============
EXCLUDE_FIELDS = {
    "chars", "sentences", "uniq_chars", "doc_id", "L1_LEX_003_top50",
    "L1_LEX_001_ttr_char",      # 字TTR（篇幅相关）
    "L1_LEX_002_ttr_word",      # 词TTR（篇幅相关：40 万字长文 0.07 vs 单篇 ~0.5；词汇丰富度改用 STTR100）
    "L2_DIS_001_avg_para_len",   # 平均段长（篇幅相关）
    "L2_DIS_001_sd_para_len",   # 段长标准差（篇幅相关）
    "L2_DIS_001_paras",          # 段落数（按篇计数，篇幅相关）
    "L1_LEX_005_dm_total",       # 旧口径：按篇原始计数（已改 dm_per_k_total）
}

# ============ 重复字段（L4 和 L1 重复，只算一次）============
DUPLICATE_FIELDS = {
    "L4_SUR_003_pos_per_k",  # 和 L1_LEX_007_pos_per_k 重复
    "L4_SUR_003_neg_per_k",  # 和 L1_LEX_007_neg_per_k 重复
    "L4_SUR_001_meta_domains",        # 根域密度之和 == L4_SUR_001_meta_per_k
    "L4_SUR_001_meta_domains_total",
}


FIELD_MAP_VERSION = 4
# v2：dm/问句/修辞反问 统一每千字；paras、meta_domains_total 不参与 Delta
# v3：词TTR 退出 Delta；我:你、正负情感比 改为 +1 平滑（零分母不再退化成原始计数）；
#     句长簇降权；基线 sd 用样本标准差（n-1）
WO_NI_CLIP = (0.1, 10.0)
# v3 口径变化的字段：旧基线（field_map_version<3）里的值不可比，baseline_stat 直接跳过
CHANGED_IN_V3 = {"L1_LEX_006_wo_ni_ratio", "L4_SUR_003_pos_neg_ratio"}
# 字段 → 口径最近一次变更的版本；基线 field_map_version 低于它时该字段跳过（口径不同不可比）
#   v3：比值 +1 平滑；v4：篇内句长 sd / burstiness 改样本标准差（n-1）
CHANGED_IN = {**{k: 3 for k in CHANGED_IN_V3},
              "L1_SYN_002_sd_sl": 4, "L1_SYN_007_burstiness": 4}

# 高相关特征组降权：组内每个特征权重 = 组权重 / 组大小（其余特征权重 1）
# 句长 6 项（均值/标准差/中位/短句/长句/CV）高度相关，原来等于给句长 6 票
FEATURE_GROUPS = {
    "sentence_length": (["L1_SYN_001_avg_sl", "L1_SYN_002_sd_sl", "L1_SYN_002_med_sl",
                         "L1_SYN_002_short_lt10", "L1_SYN_002_long_gt40", "L1_SYN_007_burstiness"], 3.0),
}
_WEIGHT = {f: w / len(fs) for fs, w in FEATURE_GROUPS.values() for f in fs}

MIN_BASELINE_DOCS = 5        # 基线最少篇数（低于此 sd 不可信）
MIN_DELTA_FEATURES = 10      # 可用（sd>0）特征最少个数


# 人读标签（迭代修正 prompt / 报告用；生成端看不懂内部字段名）
FEATURE_LABELS = {
    "L1_SYN_001_avg_sl": "平均句长（字）",
    "L1_SYN_002_sd_sl": "句长波动（标准差）",
    "L1_SYN_002_short_lt10": "短句（<10字）占比",
    "L1_SYN_002_long_gt40": "长句（>40字）占比",
    "L1_SYN_002_med_sl": "句长中位数（字）",
    "L1_SYN_007_burstiness": "句长起伏度",
    "L1_SYN_005_ba_per_k": "把字句/千字",
    "L1_SYN_005_bei_per_k": "被字句/千字",
    "L1_SYN_005_rang_per_k": "让字句/千字",
    "L1_SYN_005_shi_de_per_k": "“是…的”句/千字",
    "L1_LEX_001_sttr100": "词汇多样性（STTR）",
    "L1_LEX_007_pos_per_k": "正面情绪词/千字",
    "L1_LEX_007_neg_per_k": "负面情绪词/千字",
    "L1_SYN_006_fourchar_per_k": "四字格/千字",
    "L1_LEX_006_wo_ni_ratio": "“我”与“你”之比",
    "L2_DIS_003_pron_per_sent": "每句代词数",
    "L4_SUR_001_meta_per_k": "隐喻/千字",
    "L4_SUR_002_value_per_k": "价值判断词/千字",
    "L4_SUR_003_pos_neg_ratio": "正负情绪之比",
    "L1_SYN_004_question_per_k": "问句/千字",
    "L1_SYN_004_rhet_per_k": "反问句/千字",
    "L1_LEX_004_yqci_per_k_total": "语气词/千字",
    "L1_LEX_006_pron_per_k_total": "人称代词/千字",
    "L1_LEX_005_dm_per_k_total": "话语标记/千字",
    "L2_DIS_002_conn_per_k_total": "连接词/千字",
    "L1_LEX_008_cog_per_k_total": "认知词/千字",
    "L1_SYN_003_punct_per_k_total": "标点/千字",
}


def label(key: str) -> str:
    return FEATURE_LABELS.get(key, key)


def weight(key: str) -> float:
    """Delta 中该特征的权重（相关特征组降权）。"""
    return _WEIGHT.get(key, 1.0)


class DegenerateBaseline(ValueError):
    """基线退化（篇数太少 / sd 全 0 / 可用特征太少），Delta 不可信。"""


class EmptyText(ValueError):
    """指纹来自空文本（chars==0），不能参与 Delta。"""


def delta_features():
    """参与 Delta 的全部字段名（基线/拍平后口径）。"""
    keys = [k for k in SCALAR_FIELDS]
    keys += list(PER_DOC_FIELDS.values())
    keys += list(DICT_TO_TOTAL.values())
    return keys


def flatten(fp: dict) -> dict:
    """单篇指纹 → Delta 特征向量（全部每千字/比率口径）。唯一实现，三个脚本共用。"""
    chars = fp.get("chars") or 0
    if chars <= 0:
        raise EmptyText(f"空文本（doc_id={fp.get('doc_id', '?')}，chars=0），不能计算指纹距离")
    out = {}
    for k in SCALAR_FIELDS:
        v = fp.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if k == "L1_LEX_006_wo_ni_ratio":
                v = max(WO_NI_CLIP[0], min(v, WO_NI_CLIP[1]))
            out[k] = v
    for k, newk in PER_DOC_FIELDS.items():
        v = fp.get(k)
        if isinstance(v, (int, float)) and chars > 0:
            out[newk] = round(v * 1000 / chars, 4)
    for k, newk in DICT_TO_TOTAL.items():
        v = fp.get(k)
        if isinstance(v, dict):
            tot = sum(x for x in v.values() if isinstance(x, (int, float)))
            if k in COUNT_DICTS:
                if chars <= 0:
                    continue
                tot = tot * 1000 / chars
            out[newk] = round(tot, 4)
    return out


def baseline_stat(baseline: dict, key: str):
    """返回 (mean, sd)；兼容 v1 旧基线（按篇字段用 avg_chars 近似换算）。缺失返回 None。"""
    bl = baseline.get("baseline", baseline)
    if (baseline.get("field_map_version") or 1) < CHANGED_IN.get(key, 0):
        return None  # 旧基线该字段口径不同（未平滑 / 总体 sd），跳过
    b = bl.get(key)
    if isinstance(b, dict) and b.get("mean") is not None:
        return b["mean"], b.get("sd") or 0.0
    avg = baseline.get("avg_chars") or 0
    legacy = {v: k for k, v in PER_DOC_FIELDS.items()}
    legacy["L1_LEX_005_dm_per_k_total"] = "L1_LEX_005_dm_total"
    lk = legacy.get(key)
    if lk and avg > 0 and isinstance(bl.get(lk), dict):
        b = bl[lk]
        return b["mean"] * 1000 / avg, (b.get("sd") or 0.0) * 1000 / avg
    return None


def check_baseline(baseline: dict, min_docs: int = MIN_BASELINE_DOCS,
                   min_features: int = MIN_DELTA_FEATURES) -> int:
    """基线退化检查：篇数 < min_docs 或 sd>0 的 Delta 特征 < min_features → DegenerateBaseline。
    返回可用特征数。"""
    n_docs = baseline.get("n_docs")
    if isinstance(n_docs, int) and n_docs < min_docs:
        raise DegenerateBaseline(f"基线仅 {n_docs} 篇（< {min_docs}），sd 不可信，Delta 无意义；请用 ≥{min_docs} 篇语料建基线")
    usable = [k for k in delta_features() if (st := baseline_stat(baseline, k)) and st[1]]
    if len(usable) < min_features:
        raise DegenerateBaseline(f"基线可用特征仅 {len(usable)} 个（sd>0，需 ≥{min_features}），Delta 无意义")
    return len(usable)


def delta(fp: dict, baseline: dict, return_detail: bool = False):
    """Burrows Delta：加权 |z| 均值（相关特征组降权，见 FEATURE_GROUPS）。
    fp 为原始指纹（未拍平）。sd==0 或缺失的特征跳过；空文本抛 EmptyText。"""
    vec = flatten(fp)
    zs = {}
    for k, x in vec.items():
        st = baseline_stat(baseline, k)
        if not st or not st[1]:
            continue
        zs[k] = (x - st[0]) / st[1]
    wsum = sum(_WEIGHT.get(k, 1.0) for k in zs)
    d = sum(abs(z) * _WEIGHT.get(k, 1.0) for k, z in zs.items()) / wsum if zs else None
    if return_detail:
        return {"delta": None if d is None else round(d, 4), "n": len(zs),
                "zs": {k: round(v, 3) for k, v in zs.items()}}
    return d


def aggregate_baseline(rows: list, name: str = "", min_docs: int = MIN_BASELINE_DOCS,
                       version: str = "wfp-toolkit") -> dict:
    """逐篇指纹 → 基线 JSON（唯一实现：replicate.py 与 extract_hard_metrics --summary 共用）。
    - 收集所有行的数值字段（不只第一行）；字典字段展开 __子项 与 _total；
    - Delta 口径字段由 flatten 生成（与回测/自检同口径）；
    - sd 用样本标准差（n-1）；篇数 < min_docs 抛 DegenerateBaseline。"""
    import statistics
    rows = [r for r in rows if (r.get("chars") or 0) > 0]
    if len(rows) < min_docs:
        raise DegenerateBaseline(f"有效（非空）文档仅 {len(rows)} 篇（< {min_docs}），无法建可信基线")
    flat_rows = []
    for r in rows:
        r = dict(r)
        for k, v in list(r.items()):
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)):
                        r[f"{k}__{sk}"] = sv
                r[f"{k}_total"] = sum(x for x in v.values() if isinstance(x, (int, float)))
        r.update(flatten(r))
        flat_rows.append(r)
    keys = sorted({k for r in flat_rows for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)})
    base = {}
    for k in keys:
        vals = [r[k] for r in flat_rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            base[k] = {"mean": round(statistics.mean(vals), 4),
                       "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                       "n": len(vals)}
    avg_chars = int(round(statistics.mean(r["chars"] for r in flat_rows)))
    missing = [k for k in delta_features() if k not in base]
    out = {"n_docs": len(flat_rows), "corpus": name, "version": version,
           "avg_chars": avg_chars, "field_map_version": FIELD_MAP_VERSION,
           "missing_features": missing,
           # 建基线所用文档 id（自检用来排查留出文与基线重叠）
           "doc_ids": [str(r.get("doc_id", "")) for r in flat_rows],
           "baseline": base}
    check_baseline(out, min_docs=min_docs)
    return out


def get_all_baseline_keys():
    """返回基线里所有应该对比的字段名"""
    keys = set(SCALAR_FIELDS.keys())
    keys.update(PER_DOC_FIELDS.values())
    keys.update(DICT_TO_TOTAL.values())  # 基线里是 *_total
    return keys


def get_all_gen_keys():
    """返回生成文指纹里所有应该对比的字段名（原始名）"""
    keys = set(SCALAR_FIELDS.keys())
    keys.update(PER_DOC_FIELDS.keys())
    keys.update(DICT_TO_TOTAL.keys())  # 生成文里是原始字典名
    return keys
