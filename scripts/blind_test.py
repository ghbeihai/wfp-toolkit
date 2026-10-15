#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 盲测尺（独立信号，不并入 Delta 综合分）：
给 LLM 看目标作者的若干参考片段，再给两篇候选（A/B，一篇是作者真文、一篇是模仿文或他人文），
问哪篇更像同一作者所写。每对正反两种顺序各问一次以消除位置偏差。

输出：
  p_real_a   = LLM 认为 A 是作者真文的概率（两种顺序平均）
  pair 模式（--real/--fake）：识破率 = P(选中真文)；对模仿文而言越接近 0.5 越"以假乱真"。

用法：
  python3 blind_test.py --reference 原文目录 --real 真文.txt --fake 模仿文.txt [--n-ref 3] [--json]
  python3 blind_test.py --reference 原文目录 --pairs pairs.json   # [{"real": "...txt", "fake": "...txt"}, ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import llm_common

ROOT = Path(__file__).resolve().parent.parent
SYS = ("你是严谨的作者鉴别专家（作者鉴别任务）。根据参考片段的用词、句式、语气、论证习惯判断，"
       "不要被话题相似度影响。输出严格 JSON：{\"choice\": \"A\" 或 \"B\", \"confidence\": 0到1, \"reason\": \"一句话\"}")


def _clip(t: str, n: int) -> str:
    t = t.strip()
    return t if len(t) <= n else t[:n] + "……"


def build_messages(refs: list, a: str, b: str, ref_chars=1000, cand_chars=1500):
    ref_block = "\n\n".join(f"【参考片段 {i + 1}】\n{_clip(r, ref_chars)}" for i, r in enumerate(refs))
    user = (f"以下是同一位作者的参考片段：\n\n{ref_block}\n\n"
            f"下面两篇候选文本中，恰好有一篇出自该作者本人。\n\n【候选 A】\n{_clip(a, cand_chars)}\n\n"
            f"【候选 B】\n{_clip(b, cand_chars)}\n\n哪一篇是该作者本人写的？")
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


def judge(client, cfg, refs, real: str, fake: str) -> dict:
    """两种顺序各问一次。返回 {p_real, picks}；p_real = 选中真文的平均概率。"""
    probs, picks = [], []
    for real_first in (True, False):
        a, b = (real, fake) if real_first else (fake, real)
        d = llm_common.chat_json(client, cfg, build_messages(refs, a, b), temperature=0.0, log_prefix="[blind] ")
        ch = str(d.get("choice", "")).strip().upper()[:1]
        conf = min(max(float(d.get("confidence", 0.5)), 0.5), 1.0)  # 选中项的把握 ∈ [0.5, 1]
        picked_real = (ch == "A") == real_first
        probs.append(conf if picked_real else 1 - conf)
        picks.append({"order": "real_first" if real_first else "fake_first", "choice": ch, "picked_real": picked_real})
    return {"p_real": round(sum(probs) / len(probs), 3), "picks": picks,
            "position_consistent": picks[0]["picked_real"] == picks[1]["picked_real"]}


def load_refs(ref, n, seed=42, exclude=()):
    p = Path(ref)
    files = [p] if p.is_file() else sorted(f for f in p.iterdir() if f.suffix.lower() in (".txt", ".md"))
    files = [f for f in files if f.stem not in set(exclude)]
    if not files:
        sys.exit("错误：没有参考原文")
    pick = random.Random(seed).sample(files, min(n, len(files)))
    return [f.read_text(encoding="utf-8") for f in pick], [f.stem for f in pick]


def main():
    ap = argparse.ArgumentParser(description="LLM 盲测尺：哪篇是作者真文（正反顺序各一次）")
    ap.add_argument("--reference", required=True, help="作者参考原文目录或文件（不要包含被测真文）")
    ap.add_argument("--real", help="作者真文")
    ap.add_argument("--fake", help="模仿文 / 他人文")
    ap.add_argument("--pairs", help="批量：JSON 列表 [{real, fake}]")
    ap.add_argument("--n-ref", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not a.pairs and not (a.real and a.fake):
        ap.error("给 --real + --fake，或 --pairs")
    pairs = json.loads(Path(a.pairs).read_text(encoding="utf-8")) if a.pairs else [{"real": a.real, "fake": a.fake}]
    cfg = llm_common.load_llm_cfg(ROOT / "config" / "llm.yaml")
    client = llm_common.make_client(llm_common.require_key(ROOT), cfg)
    res = []
    for pr in pairs:
        real_t = Path(pr["real"]).read_text(encoding="utf-8")
        fake_t = Path(pr["fake"]).read_text(encoding="utf-8")
        if not real_t.strip() or not fake_t.strip():
            sys.exit(f"错误：空文本 {pr}")
        refs, ref_ids = load_refs(a.reference, a.n_ref, a.seed, exclude=[Path(pr["real"]).stem])
        try:
            r = judge(client, cfg, refs, real_t, fake_t)
        except Exception as e:  # noqa: BLE001
            r = {"p_real": None, "failed": True, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        r.update(real=pr["real"], fake=pr["fake"], refs=ref_ids)
        res.append(r)
    ok = [r["p_real"] for r in res if r.get("p_real") is not None]
    summ = {"n": len(res), "n_ok": len(ok), "mean_p_real": round(sum(ok) / len(ok), 3) if ok else None,
            "detect_rate": round(sum(p > 0.5 for p in ok) / len(ok), 3) if ok else None, "pairs": res}
    if a.json:
        print(json.dumps(summ, ensure_ascii=False, indent=2))
    else:
        for r in res:
            print(f"{Path(r['real']).name} vs {Path(r['fake']).name}: P(识破)={r.get('p_real')}")
        print(f"平均 P(识破)={summ['mean_p_real']}，识破率={summ['detect_rate']}（模仿文越接近 0.5 越以假乱真）")
    if len(ok) < len(res):
        sys.exit(3)


if __name__ == "__main__":
    main()
