#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测判别力自检：验证基线尺子能不能分清"像不像"。两关：
  关 1 留出集：原作者留出文 vs 同题对照文（control），判别比 = Δcontrol / Δheld ≥ 1.5
  关 2 异作者：原作者留出文 vs 另一作者文（other-author），判别比 = Δother / Δheld ≥ 1.2

模式：
1. 手动模式：--held-out H [--control C] [--other-author O]
   给了几个对照就跑几关；O 可为文件或目录（目录则拼接其中 .txt）。两关都给时，两关都过才 PASS。
2. 自动模式：--corpus 原作者语料目录 [--other-author 另一作者目录]
   每轮随机（--seed 可复现）留出 --holdout% 拼成留出文；有 --other-author 时跑关 2，否则同作者对照
   （同作者对照意义有限，仅作参考）。留出文若在基线语料里（baseline.doc_ids）会被剔除。

用法：
  python3 validate_discrimination.py --baseline bl.json --held-out held.txt --control ctrl.txt --other-author other.txt
  python3 validate_discrimination.py --baseline bl.json --corpus ./留出目录 --other-author ./另一作者 --rounds 5 --seed 42
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from field_map import delta as fm_delta, check_baseline, DegenerateBaseline, EmptyText

THR_CONTROL = 1.5
THR_OTHER = 1.2


def extract_fp(text_file: str) -> dict:
    extract_script = Path(__file__).parent / "extract_hard_metrics.py"
    r = subprocess.run(
        [sys.executable, str(extract_script), str(text_file), "--json"],
        capture_output=True, text=True, encoding="utf-8"
    )
    if r.returncode != 0:
        raise RuntimeError(f"指纹提取失败（{text_file}）: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout)


RULER_WEIGHTS = None  # --weights：多尺子学习权重（ruler.py learn 产出）；None = field_map 默认权重


def calc_delta(fp: dict, baseline: dict, avg_chars_override: int = None) -> float:
    """统一走 field_map.delta（与 normalize/merge 同口径）；给了 --weights 时用 ruler 综合分。
    无可用特征 → DegenerateBaseline。"""
    if avg_chars_override:
        baseline = dict(baseline, avg_chars=avg_chars_override)
    if RULER_WEIGHTS is not None:
        import ruler
        d = ruler.score(ruler.vector(fp), baseline, RULER_WEIGHTS)["composite"]
    else:
        d = fm_delta(fp, baseline)
    if d is None:
        raise DegenerateBaseline("该文本与基线没有任何可比特征（sd>0），无法计算 Delta")
    return round(d, 3)


def _ratio(d_ctrl: float, d_held: float):
    return round(d_ctrl / d_held, 2) if d_held > 0 else None


def run_one_test(held_out_file, control_file, baseline: dict, avg_chars: int = None,
                 threshold: float = THR_CONTROL) -> dict:
    """单关：held vs control（或 other）。保持旧返回结构。"""
    delta_held = calc_delta(extract_fp(held_out_file), baseline, avg_chars)
    delta_control = calc_delta(extract_fp(control_file), baseline, avg_chars)
    ratio = _ratio(delta_control, delta_held)
    return {
        "delta_held": delta_held,
        "delta_control": delta_control,
        "ratio": ratio,
        "threshold": threshold,
        "passed": ratio is not None and ratio >= threshold,
    }


def _concat_to(paths, out_path: Path) -> Path:
    out_path.write_text("\n".join(Path(p).read_text(encoding="utf-8") for p in paths), encoding="utf-8")
    return out_path


def _as_file(p: str, tmpdir: Path, tag: str) -> Path:
    """文件原样返回；目录则拼接其中全部 .txt。"""
    pp = Path(p)
    if pp.is_dir():
        files = sorted(pp.glob("*.txt"))
        if not files:
            sys.exit(f"错误：目录 {pp} 下没有 .txt")
        return _concat_to(files, tmpdir / f"_{tag}.txt")
    if not pp.exists():
        sys.exit(f"错误：文件不存在 {pp}")
    return pp


def run_manual(args, baseline, tmpdir: Path) -> dict:
    held = _as_file(args.held_out, tmpdir, "held")
    out = {"mode": "manual", "held_out": str(args.held_out)}
    d_held = calc_delta(extract_fp(held), baseline, args.avg_chars)
    out["delta_held"] = d_held
    gates = []
    if args.control:
        d = calc_delta(extract_fp(_as_file(args.control, tmpdir, "control")), baseline, args.avg_chars)
        thr = args.threshold or THR_CONTROL
        r = _ratio(d, d_held)
        out.update(delta_control=d, ratio=r, threshold=thr, passed_control=r is not None and r >= thr)
        gates.append(out["passed_control"])
    if args.other_author:
        d = calc_delta(extract_fp(_as_file(args.other_author, tmpdir, "other")), baseline, args.avg_chars)
        thr = args.threshold_other or THR_OTHER
        r = _ratio(d, d_held)
        out.update(delta_other=d, ratio_other=r, threshold_other=thr, passed_other=r is not None and r >= thr)
        gates.append(out["passed_other"])
    out["n_gates"] = len(gates)
    out["passed"] = bool(gates) and all(gates)
    return out


def run_auto(args, baseline, tmpdir: Path) -> dict:
    corpus_dir = Path(args.corpus)
    files = sorted(corpus_dir.glob("*.txt"))
    bl_ids = set(baseline.get("doc_ids") or [])
    overlap = [f for f in files if f.stem in bl_ids]
    if overlap and not args.allow_overlap:
        print(f"[warn] {len(overlap)}/{len(files)} 篇留出候选也在基线语料里（doc_ids），已剔除，"
              f"避免自己比自己（--allow-overlap 可保留）", file=sys.stderr)
        files = [f for f in files if f.stem not in bl_ids]
    if not bl_ids:
        print("[warn] 基线没有 doc_ids（旧版基线），无法检查留出文是否与基线重叠", file=sys.stderr)
    if len(files) < 10:
        sys.exit(f"错误：可用留出语料仅 {len(files)} 篇，至少 10 篇才能自动留出测试。")
    other_files = []
    if args.other_author:
        other_files = sorted(Path(args.other_author).glob("*.txt"))
        if not other_files:
            sys.exit(f"错误：--other-author 目录 {args.other_author} 下没有 .txt")
    else:
        print("[warn] 未给 --other-author：用同作者剩余文章当对照，判别比≈1 是正常的，仅作参考", file=sys.stderr)
    thr = args.threshold or (THR_OTHER if args.other_author else THR_CONTROL)
    rng = random.Random(args.seed)
    n_holdout = max(1, int(len(files) * args.holdout / 100))
    results = []
    for r in range(args.rounds):
        pool = files[:]
        rng.shuffle(pool)
        holdout = pool[:n_holdout]
        if other_files:
            ctrl = rng.sample(other_files, min(n_holdout, len(other_files)))
        else:
            ctrl = pool[n_holdout:2 * n_holdout]
        held_f = _concat_to(holdout, tmpdir / f"held_{r}.txt")
        ctrl_f = _concat_to(ctrl, tmpdir / f"ctrl_{r}.txt")
        res = run_one_test(held_f, ctrl_f, baseline, args.avg_chars, thr)
        res["held_docs"] = [f.stem for f in holdout]
        res["control_docs"] = [f.stem for f in ctrl]
        results.append(res)
        if not args.json:
            print(f"第 {r+1} 轮: 判别比 {res['ratio']} {'PASS' if res['passed'] else 'FAIL'}")
    n_pass = sum(1 for x in results if x["passed"])
    ratios = [x["ratio"] for x in results if x["ratio"] is not None]
    return {"mode": "auto", "rounds": len(results), "seed": args.seed, "threshold": thr,
            "control": "other_author" if other_files else "same_author",
            "n_pass": n_pass, "pass_rate": round(n_pass / len(results) * 100, 1),
            "avg_ratio": round(sum(ratios) / len(ratios), 2) if ratios else None,
            "passed": n_pass / len(results) >= 0.6, "results": results}


def main():
    ap = argparse.ArgumentParser(description="回测判别力自检（留出集 ≥1.5 + 异作者 ≥1.2 两关）")
    ap.add_argument("--baseline", required=True, help="基线 JSON")
    ap.add_argument("--held-out", help="原作者留出文（手动模式；文件或目录）")
    ap.add_argument("--control", help="同题对照文（手动模式关 1；文件或目录）")
    ap.add_argument("--other-author", help="另一作者文：手动模式=文件或目录（关 2）；自动模式=目录")
    ap.add_argument("--corpus", help="原作者留出语料目录（自动模式）")
    ap.add_argument("--holdout", type=int, default=20, help="自动模式每轮留出比例 %%（1-50，默认 20）")
    ap.add_argument("--rounds", type=int, default=5, help="自动模式轮数（≥1，默认 5）")
    ap.add_argument("--seed", type=int, default=42, help="自动模式随机种子（默认 42，可复现）")
    ap.add_argument("--allow-overlap", action="store_true", help="允许留出文与基线语料重叠（不推荐）")
    ap.add_argument("--avg-chars", type=int, default=None, help="旧基线按篇字段换算用的平均篇长")
    ap.add_argument("--threshold", type=float, default=None,
                    help="关 1 阈值（默认 1.5）；自动模式下作用于当前对照（默认 同作者 1.5 / 异作者 1.2）")
    ap.add_argument("--threshold-other", type=float, default=None, help="手动模式关 2 阈值（默认 1.2）")
    ap.add_argument("--min-docs", type=int, default=5, help="基线最少篇数（默认 5，低于即报错）")
    ap.add_argument("--strict", action="store_true", help="结论 FAIL 时退出码 2（CI/脚本用）")
    ap.add_argument("--weights", help="多尺子权重 JSON（ruler.py learn 产出）：用学习权重的综合分代替默认 Delta")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()
    if args.weights:
        global RULER_WEIGHTS
        RULER_WEIGHTS = json.loads(Path(args.weights).read_text(encoding="utf-8"))

    if args.rounds < 1:
        ap.error("--rounds 必须 ≥1")
    if not 1 <= args.holdout <= 50:
        ap.error("--holdout 必须在 1-50 之间")
    if not (args.held_out or args.corpus):
        ap.error("需要 --held-out（手动模式）或 --corpus（自动模式）")
    if args.held_out and not (args.control or args.other_author):
        ap.error("手动模式至少要给 --control 或 --other-author 之一")

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    try:
        check_baseline(baseline, min_docs=args.min_docs)
        with tempfile.TemporaryDirectory(prefix="wfp_vd_") as td:
            res = run_manual(args, baseline, Path(td)) if args.held_out else run_auto(args, baseline, Path(td))
    except (DegenerateBaseline, EmptyText, RuntimeError) as e:
        sys.exit(f"错误：{e}")

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif res["mode"] == "manual":
        print(f"原作者留出 Delta: {res['delta_held']}")
        if "delta_control" in res:
            print(f"关1 对照文 Delta: {res['delta_control']} → 判别比 {res['ratio']}（≥{res['threshold']}）"
                  f" {'PASS' if res['passed_control'] else 'FAIL'}")
        if "delta_other" in res:
            print(f"关2 异作者 Delta: {res['delta_other']} → 判别比 {res['ratio_other']}（≥{res['threshold_other']}）"
                  f" {'PASS' if res['passed_other'] else 'FAIL'}")
        print(f"结论（{res['n_gates']} 关）: {'PASS' if res['passed'] else 'FAIL'}")
    else:
        print(f"\n汇总：{res['rounds']} 轮（seed={res['seed']}，对照={res['control']}），"
              f"通过 {res['n_pass']} 轮，通过率 {res['pass_rate']}%")
        print(f"平均判别比: {res['avg_ratio']}")
        print(f"结论: {'PASS' if res['passed'] else 'FAIL'}（通过率 ≥60% 才算尺子可用）")
    if args.strict and not res["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
