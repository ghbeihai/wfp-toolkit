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
    "L2_DIS_001_paras": "段落数",
    "L2_DIS_003_pron_per_sent": "每句代词数",
    # L4 深层
    "L4_SUR_001_meta_per_k": "隐喻/千字",
    "L4_SUR_002_value_per_k": "价值词/千字",
    "L4_SUR_003_pos_neg_ratio": "正负情感比",
}

# ============ 按篇指标（需要按 avg_chars 换算成每千字）============
PER_DOC_FIELDS = {
    "L1_SYN_004_question_sents": "问句/篇",
    "L1_SYN_004_rhet_sents": "修辞反问/篇",
}

# ============ 字典字段（指纹是字典，基线是 *_total 标量）============
# 键 = 指纹里的原始字段名，值 = 基线里的 *_total 字段名
DICT_TO_TOTAL = {
    "L1_LEX_004_yqci_per_k": "L1_LEX_004_yqci_per_k_total",      # 语气词
    "L1_LEX_006_pron_per_k": "L1_LEX_006_pron_per_k_total",      # 人称代词
    "L1_LEX_005_dm": "L1_LEX_005_dm_total",                     # 衔接词
    "L2_DIS_002_conn_per_k": "L2_DIS_002_conn_per_k_total",     # 连接词
    "L1_LEX_008_cog_per_k": "L1_LEX_008_cog_per_k_total",       # 认知词
    "L1_SYN_003_punct_per_k": "L1_SYN_003_punct_per_k_total",     # 标点
    "L4_SUR_001_meta_domains": "L4_SUR_001_meta_domains_total", # 隐喻根域
}

# ============ 排除字段（篇幅量 / 重复，不参与 Delta）============
EXCLUDE_FIELDS = {
    "chars", "sentences", "uniq_chars", "doc_id", "L1_LEX_003_top50",
    "L1_LEX_001_ttr_char",      # 字TTR（篇幅相关）
    "L2_DIS_001_avg_para_len",   # 平均段长（篇幅相关）
    "L2_DIS_001_sd_para_len",   # 段长标准差（篇幅相关）
}

# ============ 重复字段（L4 和 L1 重复，只算一次）============
DUPLICATE_FIELDS = {
    "L4_SUR_003_pos_per_k",  # 和 L1_LEX_007_pos_per_k 重复
    "L4_SUR_003_neg_per_k",  # 和 L1_LEX_007_neg_per_k 重复
}


def get_all_baseline_keys():
    """返回基线里所有应该对比的字段名"""
    keys = set(SCALAR_FIELDS.keys())
    keys.update(PER_DOC_FIELDS.keys())
    keys.update(DICT_TO_TOTAL.values())  # 基线里是 *_total
    return keys


def get_all_gen_keys():
    """返回生成文指纹里所有应该对比的字段名（原始名）"""
    keys = set(SCALAR_FIELDS.keys())
    keys.update(PER_DOC_FIELDS.keys())
    keys.update(DICT_TO_TOTAL.keys())  # 生成文里是原始字典名
    return keys
