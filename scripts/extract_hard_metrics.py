#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WFP · 文风指纹 — L1/L2 硬指标提取（v0.2 全量）

学术根基（完整谱系与核验状态见 docs/WFP学术基础研究.md）：
  L1_LEX_001/002 TTR/STTR —— Holmes 1994（滑动窗口是长文本标准解法）
  L1_LEX_003 高频词画像    —— Mosteller & Wallace 1964；Burrows 2002（Delta）
  L1_LEX_004 语气词        —— 方梅 2000（《中国语文》2000(5):459-470）
  L1_LEX_005 话语标记      —— Schiffrin 1987；方梅 2000
  L1_LEX_006 人称代词      —— Pennebaker/LIWC；Park et al. 2015
  L1_LEX_007 情感词        —— LIWC（Pennebaker et al. 2015）；知网 HowNet 情感词典
  L1_LEX_008 认知词        —— LIWC 认知过程维度；LIWC-22（Boyd et al. 2022）
  L1_SYN_001/002 句长      —— Yule 1939；Morton 1978；Grieve 2007
  L1_SYN_003 标点          —— Baayen, van Halteren & Tweedie 1996
  L1_SYN_004 设问/反问     —— 陈望道《修辞学发凡》1932（设问/反问格）
  L1_SYN_005 固定句式      —— 朱德熙《语法讲义》1982（把字句/被字句/判断句）
  L1_SYN_006 四字格        —— 刘叔新《汉语描写词汇学》1990（熟语与固定语）
  L2_DIS_001 段落结构      —— Brown & Yule 1983；Halliday 1994
  L2_DIS_002 衔接词        —— Halliday & Hasan 1976（衔接理论）
  L2_DIS_003 指代回指      —— Halliday & Hasan 1976；Ariel 1990（可及性理论）

操作化口径（与文献来源分开标注，属工程定义）：
  情感/认知词表为 v0.2 初步词表（按 LIWC 维度/HowNet 口径整理），v0.3 接入正式词典；
  句式/四字格用正则近似，v0.3 可换句法标注结果；指代回指为代词密度代理（半自动）。

用法:
  python3 scripts/extract_hard_metrics.py <file> [--json]            # 单文本
  python3 scripts/extract_hard_metrics.py --corpus-spi [--limit N]   # SPI 431 篇（篇章级）
  python3 scripts/extract_hard_metrics.py --corpus-aot [--limit N]   # AoT 块级
  python3 scripts/extract_hard_metrics.py --summary <glob>           # 聚合基线(mean/std)
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter

PUNCT_ORDER = ["，", "。", "！", "？", "、", "；", "：", "——", "……", "“", "”"]
YQCI = ["啊", "呢", "吧", "嘛", "哦", "呀", "啦", "哈", "呗", "么"]
PRONOUNS = ["我", "你", "他", "她", "我们", "你们", "咱们"]
DISCOURSE_MARKERS = ["这个", "就是说", "然后", "其实", "反正", "是不是", "对不对", "有没有", "怎么说呢", "你看"]
CONNECTIVES = {
    "转折": ["但是", "然而", "不过", "可是"],
    "因果": ["所以", "因此", "因为"],
    "递进": ["而且", "并且"],
    "时序": ["然后", "接着", "最后"],
}
RHET_QUEST_MARKS = ["难道", "岂", "怎么", "难道不是", "岂不", "何尝"]
# 无问号疑问结构（口语附加问/疑问词）：是不是/对不对/有没有 等
_IDIOM_SET = None
def _load_idioms():
    """加载 THUOCL 成语词表（清华开放中文词库）——L1_SYN_006 口径词表"""
    global _IDIOM_SET
    if _IDIOM_SET is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "thuocl_chengyu.txt")
        _IDIOM_SET = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.split()[0].strip()
                if len(w) == 4:
                    _IDIOM_SET.add(w)
    return _IDIOM_SET

QUESTION_STRUCT = ["是不是", "对不对", "有没有", "能不能", "行不行", "要不要", "会不会",
                   "愿不愿意", "为什么", "凭什么", "怎么会", "难道", "岂", "何尝", "哪里", "什么", "啥", "吗"]
QUESTION_END_PARTICLE = ("吗", "么", "呢")
# L4 隐喻根域词元（操作化 v0.4）：根域名冻结自 AoT尺子_v1.0 维度6（12 根域）；
# 词元为 Lakoff & Johnson 1980 概念隐喻的典型词汇化表达（旅程隐喻/战争隐喻…），工程口径，来源见 docs/WFP学术基础研究.md 第 8 章
METAPHOR_DOMAINS = {
    "旅程": ["走", "路", "前行", "方向", "终点", "起点", "驿站", "迷路", "跨越", "旅途", "脚步", "到达"],
    "战争": ["战斗", "打赢", "打输", "敌人", "进攻", "防守", "战场", "胜利", "失败", "攻克", "搏斗", "武器"],
    "建筑": ["建", "结构", "地基", "框架", "支柱", "搭建", "倒塌", "支撑", "基础", "大厦", "盖楼", "加固"],
    "机器": ["运转", "零件", "齿轮", "引擎", "动力", "卡住", "系统", "机制", "启动", "运作", "链条", "轴承"],
    "生命": ["成长", "发育", "种子", "根", "开花", "结果", "养分", "生长", "生命力", "发芽", "枯萎", "繁殖"],
    "市场": ["价值", "成本", "价格", "买卖", "投资", "回报", "交易", "利润", "划算", "亏本", "行情", "筹码"],
    "游戏": ["规则", "下棋", "对手", "一局", "游戏", "赌注", "输赢", "博弈", "出牌", "套路", "局", "玩家"],
    "剧场": ["舞台", "登场", "扮演", "角色", "剧本", "谢幕", "观众", "演戏", "台前", "幕后", "布景", "主角"],
    "容器": ["里面", "外面", "装满", "充满", "溢出", "容纳", "放进", "倒出", "容器", "装进", "掏空", "塞满"],
    "光暗": ["光明", "黑暗", "照亮", "阴影", "阳光", "乌云", "明亮", "暗淡", "曙光", "迷雾", "曝光", "阴霾"],
    "上下": ["上升", "下降", "提高", "跌落", "顶层", "底层", "向上", "往下", "攀上", "坠下", "高低", "起伏"],
    "温度": ["热", "冷", "温暖", "冰冷", "温度", "暖", "凉", "寒", "发热", "降温", "炽热", "寒意"],
}
VALUE_WORDS = ["应该", "必须", "值得", "重要", "最好", "一定", "千万", "切记", "务必", "别忘"]
COGNITIVE_WORDS = ["觉得", "认为", "可能", "应该", "相信", "明白", "知道", "想", "理解", "似乎", "大概", "肯定", "必须", "记得", "考虑"]
# 操作化初步词表（v0.2）：按 LIWC 2015 中文情感维度 / 知网 HowNet 情感分析口径整理；正式词典 v0.3 接入
POS_WORDS = ["好", "棒", "厉害", "优秀", "健康", "幸福", "开心", "喜欢", "爱", "美", "善", "真", "重要", "有效", "安全", "靠谱", "放心", "希望", "成功", "稳定", "温暖", "感谢", "坚持", "努力", "进步", "强大", "信任", "赞同", "正确", "有益"]
NEG_WORDS = ["坏", "差", "糟糕", "可怕", "危险", "痛苦", "难过", "讨厌", "恨", "丑", "恶", "假", "无效", "不安全", "骗", "担心", "恐惧", "失败", "崩溃", "寒冷", "指责", "放弃", "退步", "虚弱", "怀疑", "反对", "错误", "有害", "麻烦", "焦虑"]


def clean_text(raw: str) -> str:
    """去空白；剥离 markdown 标题行（# 开头）噪音。"""
    lines = []
    for ln in raw.splitlines():
        if re.match(r"^#+\s", ln):
            continue
        lines.append(ln)
    return re.sub(r"\s+", "", "\n".join(lines))


def split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"[。！？!?…]+", text) if s.strip()]


def split_paragraphs(raw: str) -> list:
    paras = [re.sub(r"\s+", "", p).strip() for p in re.split(r"\n", raw) if p.strip()]
    return [p for p in paras if p]


_JIEBA_WARNED = False


def per_k(chars: int, n: int) -> float:
    return round(n * 1000.0 / max(chars, 1), 3)


def sttr(text: str, window: int = 100, step: int = 50) -> float:
    """滑动窗口 STTR（Holmes 1994 长文本标准做法）。"""
    vals = []
    i = 0
    while i + window <= len(text):
        w = text[i:i + window]
        vals.append(len(set(w)) / window)
        i += step
    return round(statistics.mean(vals), 4) if vals else None  # 不足一个窗口：无法计算


def extract_metrics(raw_text: str, doc_id: str = "") -> dict:
    text = clean_text(raw_text)
    paras = split_paragraphs(raw_text)
    sents = split_sentences(text)

    chars = len(text)
    uniq_chars = len(set(text))
    ttr_char = round(uniq_chars / max(chars, 1), 4)
    sttr_val = sttr(text)

    # 词级（jieba 可选）
    ttr_word = None
    top_words = []
    words = None  # 无 jieba 时为 None：依赖分词的指标输出 None，而不是静默 0
    try:
        import jieba
    except ImportError:
        global _JIEBA_WARNED
        if not _JIEBA_WARNED:
            print("[warn] 未安装 jieba：词TTR/四字格 输出 None（pip install jieba）", file=sys.stderr)
            _JIEBA_WARNED = True
    else:
        words = [w for w in jieba.cut(text) if w.strip() and w not in PUNCT_ORDER]
        if words:
            ttr_word = round(len(set(words)) / len(words), 4)
        top_words = Counter(words).most_common(50)

    slen = [len(s) for s in sents]
    avg_sl = round(statistics.mean(slen), 2) if slen else None
    sd_sl = round(statistics.pstdev(slen), 2) if len(slen) > 1 else 0.0
    # L1_SYN_007 burstiness 句长变异系数（CV=sd/mean）：人类文本通常 >0.5，机器文本 <0.3
    burstiness = round(sd_sl / avg_sl, 4) if avg_sl else None
    short_pct = round(sum(1 for n in slen if n < 10) / max(len(slen), 1), 4)
    long_pct = round(sum(1 for n in slen if n > 40) / max(len(slen), 1), 4)
    med_sl = round(statistics.median(slen), 2) if slen else None

    para_lens = [len(p) for p in paras]
    avg_pl = round(statistics.mean(para_lens), 2) if para_lens else 0.0
    sd_pl = round(statistics.pstdev(para_lens), 2) if len(para_lens) > 1 else 0.0

    # 情感词密度（操作化词表）
    pos_n = sum(text.count(w) for w in POS_WORDS)
    neg_n = sum(text.count(w) for w in NEG_WORDS)

    # 设问/反问（陈望道 1932）：问句 = 问号句 + 无问号的疑问结构句（口语附加问/疑问词/短句疑问语气词）
    # 注：很多人说问句不加问号（对不对/是不是/有没有 结尾），故不能只数问号
    q_mark = text.count("？")
    q_struct = sum(1 for s in sents
                   if any(m in s for m in QUESTION_STRUCT)
                   or (len(s) <= 20 and s.endswith(QUESTION_END_PARTICLE)))
    question_sents = q_mark + q_struct
    rhet_quest_sents = sum(1 for s in sents if any(m in s for m in RHET_QUEST_MARKS))

    # 固定句式（朱德熙 1982；正则近似）
    ba_sent = len(re.findall(r"把[^，。！？；]{1,12}", text))
    bei_sent = len(re.findall(r"被[^，。！？；]{1,12}", text))
    rang_sent = len(re.findall(r"让[^，。！？；]{1,12}", text))
    shi_de_sent = len(re.findall(r"是[^，。！？；]{1,20}的", text))

    # 四字格密度（刘叔新 1990；操作化口径：jieba 分词 ∩ THUOCL 成语词表）
    _idioms = _load_idioms()
    four_char = sum(1 for w in words if w in _idioms) if words is not None else None

    # 指代回指代理（Halliday & Hasan 1976；Ariel 1990；代词密度 + 句均代词）
    pronoun_total = sum(text.count(p) for p in PRONOUNS)

    # L4 表层投影（只测不释；构念解释归 SPI/AoT）
    meta_domains = {d: sum(text.count(w) for w in ws) for d, ws in METAPHOR_DOMAINS.items()}
    meta_total = sum(meta_domains.values())
    value_n = sum(text.count(w) for w in VALUE_WORDS)

    res = {
        "doc_id": doc_id or "unnamed",
        "chars": chars,
        "uniq_chars": uniq_chars,
        "L1_LEX_001_ttr_char": ttr_char,
        "L1_LEX_001_sttr100": sttr_val,
        "L1_LEX_002_ttr_word": ttr_word,
        "L1_LEX_003_top50": [{"w": w, "n": n} for w, n in top_words],
        "L1_LEX_004_yqci_per_k": {y: per_k(chars, text.count(y)) for y in YQCI},
        "L1_LEX_005_dm": {m: text.count(m) for m in DISCOURSE_MARKERS},
        "L1_LEX_006_pron_per_k": {p: per_k(chars, text.count(p)) for p in PRONOUNS},
        "L1_LEX_006_wo_ni_ratio": (round(text.count("我") / max(text.count("你"), 1), 3)
                                   if (text.count("我") or text.count("你")) else None),
        "L1_LEX_007_pos_per_k": per_k(chars, pos_n),
        "L1_LEX_007_neg_per_k": per_k(chars, neg_n),
        "L1_LEX_008_cog_per_k": {w: per_k(chars, text.count(w)) for w in COGNITIVE_WORDS},
        "L1_SYN_001_avg_sl": avg_sl,
        "L1_SYN_002_sd_sl": sd_sl,
        "L1_SYN_002_med_sl": med_sl,
        "L1_SYN_002_short_lt10": short_pct,
        "L1_SYN_002_long_gt40": long_pct,
        "L1_SYN_003_punct_per_k": {p: per_k(chars, text.count(p)) for p in PUNCT_ORDER},
        "L1_SYN_004_question_sents": question_sents,
        "L1_SYN_004_rhet_sents": rhet_quest_sents,
        "L1_SYN_005_ba_per_k": per_k(chars, ba_sent),
        "L1_SYN_005_bei_per_k": per_k(chars, bei_sent),
        "L1_SYN_005_rang_per_k": per_k(chars, rang_sent),
        "L1_SYN_005_shi_de_per_k": per_k(chars, shi_de_sent),
        "L1_SYN_006_fourchar_per_k": per_k(chars, four_char) if four_char is not None else None,
        "L1_SYN_007_burstiness": burstiness,
        "L2_DIS_001_paras": len(paras),
        "L2_DIS_001_avg_para_len": avg_pl,
        "L2_DIS_001_sd_para_len": sd_pl,
        "L2_DIS_002_conn_per_k": {c: per_k(chars, sum(text.count(w) for w in ws)) for c, ws in CONNECTIVES.items()},
        "L2_DIS_003_pron_per_sent": round(pronoun_total / max(len(sents), 1), 3),
        "L4_SUR_001_meta_per_k": per_k(chars, meta_total),
        "L4_SUR_001_meta_domains": {d: per_k(chars, n) for d, n in meta_domains.items()},
        "L4_SUR_002_value_per_k": per_k(chars, value_n),
        "L4_SUR_003_pos_per_k": per_k(chars, pos_n),
        "L4_SUR_003_neg_per_k": per_k(chars, neg_n),
        "L4_SUR_003_pos_neg_ratio": round((pos_n / max(neg_n, 1)), 3),
        "sentences": len(sents),
    }
    return res


def summarize(files: list) -> dict:
    rows = [json.load(open(f, encoding="utf-8")) for f in files]
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    base = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and r[k] is not None]
        if vals:
            base[k] = {
                "mean": round(statistics.mean(vals), 4),
                "sd": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        # 句数分层基线（v0.4.19.1）：burstiness 存在短文 CV 系统性偏低，
        # 按句数档位单独聚合，供 delta() 按样本句数自动选档
        if k == "L1_SYN_007_burstiness":
            short_vals = [r[k] for r in rows
                          if isinstance(r.get(k), (int, float)) and r[k] is not None
                          and 15 <= r.get("sentences", 0) <= 45]
            long_vals = [r[k] for r in rows
                         if isinstance(r.get(k), (int, float)) and r[k] is not None
                         and r.get("sentences", 0) >= 50]
            if short_vals:
                base[f"{k}__short"] = {
                    "mean": round(statistics.mean(short_vals), 4),
                    "sd": round(statistics.pstdev(short_vals), 4) if len(short_vals) > 1 else 0.0,
                    "n": len(short_vals),
                }
            if long_vals:
                base[f"{k}__long"] = {
                    "mean": round(statistics.mean(long_vals), 4),
                    "sd": round(statistics.pstdev(long_vals), 4) if len(long_vals) > 1 else 0.0,
                    "n": len(long_vals),
                }
    return {"n_docs": len(rows), "baseline": base}


def main() -> None:
    ap = argparse.ArgumentParser(description="WFP L1/L2 硬指标提取 v0.2")
    ap.add_argument("file", nargs="?", help="UTF-8 文本文件路径")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    ap.add_argument("--corpus-spi", action="store_true", help="SPI 431 篇全量（篇章级）")
    ap.add_argument("--corpus-aot", action="store_true", help="AoT 块级全量")
    ap.add_argument("--limit", type=int, default=0, help="限制处理篇数（0=全量）")
    ap.add_argument("--summary", nargs="+", help="聚合指纹 JSON 生成基线")
    args = ap.parse_args()

    if args.summary:
        files = []
        for g in args.summary:
            files.extend(glob.glob(g))
        base = summarize(files)
        print(json.dumps(base, ensure_ascii=False, indent=2))
        return

    # v0.4.20：数据源从 config/data_sources.yaml 读取（CLI 参数可覆盖）
    import yaml as _yaml
    WFP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ds = _yaml.safe_load(open(os.path.join(WFP_ROOT, "config/data_sources.yaml"), encoding="utf-8"))
    SPI_PATH = getattr(args, "spi_path", None) or _ds["corpus"]["spi_articles"]
    AOT_DIR = getattr(args, "aot_dir", None) or _ds["corpus"]["aot_dir"]
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results", "fingerprints")
    os.makedirs(OUT_DIR, exist_ok=True)

    def run_batch(items, out_prefix):
        n = 0
        for doc_id, text in items:
            if args.limit and n >= args.limit:
                break
            m = extract_metrics(text, doc_id)
            safe = re.sub(r"[^\w\u4e00-\u9fa5-]", "_", doc_id)[:60]
            with open(f"{OUT_DIR}/{out_prefix}_{safe}.json", "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False)
            n += 1
        print(f"完成 {n} 篇 -> {OUT_DIR}/{out_prefix}_*.json")

    if args.corpus_spi:
        d = json.load(open(SPI_PATH, encoding="utf-8"))
        items = []
        for title, v in d.items():
            txt = " ".join(v) if isinstance(v, list) else str(v)
            items.append((title, txt))
        run_batch(items, "spi")
        return

    if args.corpus_aot:
        total = 0
        for fn in sorted(os.listdir(AOT_DIR)):
            if not fn.endswith(".json"):
                continue
            d = json.load(open(os.path.join(AOT_DIR, fn), encoding="utf-8"))
            items = d.get("items", [])
            n = 0
            for it in items:
                if args.limit and total >= args.limit:
                    break
                title = it.get("title", "") or it.get("id", "")
                txt = it.get("text") or it.get("content") or ""
                if not txt:
                    continue
                m = extract_metrics(txt, title)
                # 文件名用唯一块 id，避免同标题多块互相覆盖
                safe = re.sub(r"[^\w\u4e00-\u9fa5-]", "_", f"{fn[:-5]}_{it.get('id','')}")[:60]
                with open(f"{OUT_DIR}/aot_{safe}.json", "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False)
                n += 1
                total += 1
            print(f"{fn}: {n} 块")
        print(f"AoT 合计 {total} 块")
        return

    if not args.file:
        ap.error("需要 <file> 或 --corpus-spi / --corpus-aot")

    raw = open(args.file, encoding="utf-8").read()
    doc_id = os.path.splitext(os.path.basename(args.file))[0]
    res = extract_metrics(raw, doc_id)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    print(f"[{res['doc_id']}]")
    print(f"  字符 {res['chars']} | 句子 {res['sentences']} | 段落 {res['L2_DIS_001_paras']} | "
          f"平均句长 {res['L1_SYN_001_avg_sl']} (σ{res['L1_SYN_002_sd_sl']}) | 段长 {res['L2_DIS_001_avg_para_len']} | "
          f"burstiness {res['L1_SYN_007_burstiness']}")
    print(f"  字符TTR {res['L1_LEX_001_ttr_char']} | STTR100 {res['L1_LEX_001_sttr100']} | "
          f"词TTR {res['L1_LEX_002_ttr_word'] if res['L1_LEX_002_ttr_word'] is not None else 'N/A'}")
    print(f"  短句<10字 {res['L1_SYN_002_short_lt10']*100:.1f}% | 长句>40字 {res['L1_SYN_002_long_gt40']*100:.1f}% | "
          f"问句 {res['L1_SYN_004_question_sents']} | 反问标记 {res['L1_SYN_004_rhet_sents']}")
    print(f"  情感词 +{res['L1_LEX_007_pos_per_k']}/千字 -{res['L1_LEX_007_neg_per_k']}/千字 | "
          f"我:你 {res['L1_LEX_006_wo_ni_ratio']} | 句均代词 {res['L2_DIS_003_pron_per_sent']}")
    yq = {k: v for k, v in res["L1_LEX_004_yqci_per_k"].items() if v > 0}
    conn = {k: v for k, v in res["L2_DIS_002_conn_per_k"].items() if v > 0}
    if yq:
        print("  语气词/千字:", yq)
    if conn:
        print("  衔接词/千字:", conn)
    print("  句式/千字 把", res['L1_SYN_005_ba_per_k'], "被", res['L1_SYN_005_bei_per_k'],
          "让", res['L1_SYN_005_rang_per_k'], "是…的", res['L1_SYN_005_shi_de_per_k'],
          "| 四字格", res['L1_SYN_006_fourchar_per_k'])


if __name__ == "__main__":
    main()
