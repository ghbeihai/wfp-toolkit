#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体裁自动识别 + 语料门控（corpus gate）。

体裁：lecture（讲座/讲课/口播转写）、essay（随笔/科普/评论文章）、news（新闻/通稿）、
      dialogue（对话/访谈/问答实录）、poetry（诗歌/歌词）、other。
- 规则启发式（默认，零成本）：口语标记密度、说话人行、新闻要素、分行短句等；
- 可选 LLM（--llm）：规则置信度 < --llm-below 时再问 LLM（走 llm_common：缓存/退避/mock）。

门控（gate）：
- 某体裁组 < min_docs 篇 → 该组只做分析（analysis_only），不建基线、不做生成/回测；
- 最大体裁占比 < mixed_share（默认 0.6）→ 语料混杂（mixed），整体基线只做分析并给出明确警告。

用法：
  python3 genre.py 文章.txt [文章2.txt ...] [--llm] [--json]
  python3 genre.py --dir 原文目录 --json        # 输出逐篇体裁 + 分组 + 门控结论
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

GENRES = ("lecture", "essay", "news", "dialogue", "poetry", "other")
GENRE_ZH = {"lecture": "讲座/口播转写", "essay": "随笔/文章", "news": "新闻", "dialogue": "对话/访谈",
            "poetry": "诗歌/歌词", "other": "其他"}

_ORAL = re.compile(r"大家|各位|同学们|朋友们|我们今天|今天我们|那么|对吧|是吧|好不好|对不对|你看|咱们|啊|呢|嘛|吧")
_NEWS = re.compile(r"记者|据.{0,6}(报道|消息)|日电|通讯员|新华社|发布会|据悉|表示|截至|同比|环比")
_DATE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日")
_SPEAKER = re.compile(r"^\s*([A-Za-z\u4e00-\u9fa5]{1,6}|[问答主持人记者])[:：]", re.M)
_QA = re.compile(r"^\s*[问答Qq][:：]", re.M)


def features(text: str) -> dict:
    t = text.strip()
    chars = len(re.sub(r"\s+", "", t)) or 1
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    short_lines = sum(1 for x in lines if len(x) <= 16 and not re.search(r"[。！？；]$", x[-1:] or ""))
    return {
        "chars": chars,
        "oral_per_k": len(_ORAL.findall(t)) * 1000 / chars,
        "news_hits": len(_NEWS.findall(t)) + len(_DATE.findall(t)),
        "speaker_lines": len(_SPEAKER.findall(t)) + len(_QA.findall(t)),
        "n_lines": len(lines),
        "short_line_share": short_lines / max(len(lines), 1),
        "avg_line": chars / max(len(lines), 1),
        "quote_per_k": (t.count("“") + t.count("「")) * 1000 / chars,
    }


def classify_rule(text: str) -> dict:
    """返回 {genre, confidence, scores, method:"rule"}。"""
    f = features(text)
    sc = {g: 0.0 for g in GENRES}
    if f["n_lines"] >= 4 and f["short_line_share"] >= 0.6 and f["avg_line"] <= 16:
        sc["poetry"] += 2.0 + f["short_line_share"]
    sl = f["speaker_lines"]
    if sl >= 3 and sl / max(f["n_lines"], 1) >= 0.25:
        sc["dialogue"] += 2.0 + min(sl / max(f["n_lines"], 1), 1.0)
    if f["news_hits"] >= 3:
        sc["news"] += 1.0 + min(f["news_hits"] / 6, 1.5)
    if f["oral_per_k"] >= 12:
        sc["lecture"] += 1.0 + min((f["oral_per_k"] - 12) / 12, 1.5)
    sc["essay"] += 1.0  # 默认先验：普通文章
    best = max(sc, key=sc.get)
    srt = sorted(sc.values(), reverse=True)
    conf = round(min(0.95, 0.5 + (srt[0] - srt[1]) / 3), 2)
    return {"genre": best, "confidence": conf, "scores": {k: round(v, 2) for k, v in sc.items() if v},
            "features": {k: round(v, 2) for k, v in f.items()}, "method": "rule"}


_LLM_SYS = ("你是中文文本体裁分类器。体裁只能是：lecture（讲座/讲课/口播转写）、essay（随笔/科普/评论文章）、"
            "news（新闻/通稿）、dialogue（对话/访谈/问答实录）、poetry（诗歌/歌词）、other。"
            '输出严格 JSON：{"genre": "<体裁>", "confidence": 0到1}')


def classify_llm(text: str, client=None, cfg=None) -> dict:
    import llm_common
    root = Path(__file__).resolve().parent.parent
    cfg = cfg or llm_common.load_llm_cfg(root / "config" / "llm.yaml")
    client = client or llm_common.make_client(llm_common.require_key(root), cfg)
    t, _ = llm_common.truncate(text, dict(cfg, max_text_chars=2000))
    d = llm_common.chat_json(client, cfg, [{"role": "system", "content": _LLM_SYS},
                                           {"role": "user", "content": f"文本：\n{t}"}],
                             temperature=0.0, log_prefix="[genre] ")
    g = d.get("genre") if d.get("genre") in GENRES else "other"
    return {"genre": g, "confidence": float(d.get("confidence", 0.5)), "method": "llm"}


def classify(text: str, use_llm: bool = False, llm_below: float = 0.7, _ctx: dict | None = None) -> dict:
    r = classify_rule(text)
    if use_llm and r["confidence"] < llm_below:
        try:
            ctx = _ctx if _ctx is not None else {}
            if "client" not in ctx:
                import llm_common
                root = Path(__file__).resolve().parent.parent
                ctx["cfg"] = llm_common.load_llm_cfg(root / "config" / "llm.yaml")
                ctx["client"] = llm_common.make_client(llm_common.require_key(root), ctx["cfg"])
            l = classify_llm(text, ctx["client"], ctx["cfg"])
            r = dict(r, genre=l["genre"], confidence=l["confidence"], method="rule+llm", rule_genre=r["genre"])
        except Exception as e:  # noqa: BLE001 —— LLM 失败退回规则结果
            print(f"[warn] 体裁 LLM 判定失败，沿用规则结果：{e}", file=sys.stderr)
    return r


def gate(labels: dict, min_docs: int = 5, mixed_share: float = 0.6) -> dict:
    """labels: {doc_id: genre} → {groups, dominant, share, status, usable_genres, warnings}"""
    groups = {}
    for d, g in labels.items():
        groups.setdefault(g, []).append(d)
    n = len(labels)
    dom = max(groups, key=lambda g: len(groups[g])) if groups else None
    share = len(groups[dom]) / n if n else 0.0
    warnings, usable = [], []
    for g, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(ids) < min_docs:
            warnings.append(f"体裁「{GENRE_ZH.get(g, g)}」仅 {len(ids)} 篇（< {min_docs}）：只做分析，不建基线、不用于生成/回测")
        else:
            usable.append(g)
    mixed = share < mixed_share
    if mixed:
        warnings.append(f"语料体裁混杂：最大体裁「{GENRE_ZH.get(dom, dom)}」只占 {share:.0%}（< {mixed_share:.0%}），"
                        "整体基线只做分析；生成/回测请按体裁分组基线")
    if not usable:
        warnings.append("没有任何体裁组达到最小篇数：整份语料只做分析（analysis_only），Delta/判别比不可信")
    status = "ok" if usable and not mixed else ("mixed" if usable else "analysis_only")
    return {"n_docs": n, "groups": {g: sorted(v) for g, v in groups.items()},
            "dominant": dom, "dominant_share": round(share, 3), "status": status,
            "usable_genres": usable, "warnings": warnings, "min_docs": min_docs, "mixed_share": mixed_share}


def main():
    ap = argparse.ArgumentParser(description="体裁识别 + 语料门控")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir", help="原文目录（.txt/.md）")
    ap.add_argument("--llm", action="store_true", help="规则置信度低时用 LLM 复判（需 key 或 WENBI_MOCK_LLM）")
    ap.add_argument("--llm-below", type=float, default=0.7)
    ap.add_argument("--min-docs", type=int, default=5)
    ap.add_argument("--mixed-share", type=float, default=0.6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    files = [Path(f) for f in args.files]
    if args.dir:
        files += sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in (".txt", ".md"))
    if not files:
        ap.error("给文件或 --dir")
    ctx, per = {}, {}
    for f in files:
        t = f.read_text(encoding="utf-8")
        if not t.strip():
            print(f"[warn] 跳过空文本 {f}", file=sys.stderr)
            continue
        per[f.stem] = classify(t, args.llm, args.llm_below, ctx)
    g = gate({k: v["genre"] for k, v in per.items()}, args.min_docs, args.mixed_share)
    if args.json:
        print(json.dumps({"docs": per, "gate": g}, ensure_ascii=False, indent=2))
        return
    for k, v in per.items():
        print(f"{k}\t{v['genre']}\t{v['confidence']}\t{v['method']}")
    print(f"门控：{g['status']}（主体裁 {g['dominant']} {g['dominant_share']:.0%}）")
    for w in g["warnings"]:
        print(f"[警告] {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
