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
    "L1_LEX_002_ttr_word": "词TTR",
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


FIELD_MAP_VERSION = 2   # v2：dm/问句/修辞反问 统一每千字；paras、meta_domains_total 不参与 Delta
WO_NI_CLIP = (0.1, 10.0)


def delta_features():
    """参与 Delta 的全部字段名（基线/拍平后口径）。"""
    keys = [k for k in SCALAR_FIELDS]
    keys += list(PER_DOC_FIELDS.values())
    keys += list(DICT_TO_TOTAL.values())
    return keys


def flatten(fp: dict) -> dict:
    """单篇指纹 → Delta 特征向量（全部每千字/比率口径）。唯一实现，三个脚本共用。"""
    chars = fp.get("chars") or 0
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


def delta(fp: dict, baseline: dict, return_detail: bool = False):
    """Burrows Delta：|z| 均值。fp 为原始指纹（未拍平）。sd==0 或缺失的特征跳过。"""
    vec = flatten(fp)
    zs = {}
    for k, x in vec.items():
        st = baseline_stat(baseline, k)
        if not st or not st[1]:
            continue
        zs[k] = (x - st[0]) / st[1]
    d = sum(abs(z) for z in zs.values()) / len(zs) if zs else None
    if return_detail:
        return {"delta": None if d is None else round(d, 4), "n": len(zs),
                "zs": {k: round(v, 3) for k, v in zs.items()}}
    return d


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
