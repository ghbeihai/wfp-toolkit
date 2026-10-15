#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4 隐喻识别（DeepSeek 版）

用法：
  python3 score_metaphor.py 某篇文章.txt --json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

METAPHOR_DOMAINS = {
    "旅程": "把人生/事业/关系比作旅程",
    "战争": "把争论/竞争/对抗比作战争",
    "建筑": "把体系/人格/作品比作建筑",
    "机器": "把组织/身体/社会比作机器",
    "生命": "把抽象事物比作生物",
    "市场": "把关系/能力/事物比作商品交易",
    "游戏": "把生活/竞争比作游戏博弈",
    "剧场": "把社会/人生比作舞台戏剧",
    "容器": "把抽象状态比作容器",
    "光暗": "把情绪/认知比作光暗",
    "上下": "把地位/数量/情绪比作垂直方位",
    "温度": "把关系/情绪比作温度",
}

SYSTEM_PROMPT = """你是一位严谨的隐喻识别编码员。

你会收到一段中文文本。任务：通读全文，找出所有【概念隐喻】——即把抽象事物比作具体事物的表达。

输出严格 JSON，无其他文字：
{
  "metaphors": [
    {"text": "原文片段（逐字引用）", "domain": "12根域之一", "explain": "为什么是这个域，20字内"}
  ]
}

12 个根域只能是：旅程、战争、建筑、机器、生命、市场、游戏、剧场、容器、光暗、上下、温度。
找不到任何隐喻时输出 {"metaphors": []}。"""

CHUNK_SIZE = 2000  # 分块阈值：超过 2000 字就分块


def load_key() -> str | None:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def count_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def split_text(text: str, max_len: int = CHUNK_SIZE) -> list[str]:
    """按段落分块，每块不超过 max_len 字"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > max_len and current:
            chunks.append(current)
            current = p
        else:
            current = current + "\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


def score_chunk(text: str, client, model: str, retries: int = 2):
    """返回该块的隐喻列表；被截断（finish_reason=length）或 JSON 解析失败时重试，
    仍失败返回 None（调用方计入 failed_chunks，而不是静默当作 0 个隐喻）。"""
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=8192,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        truncated = getattr(choice, "finish_reason", None) == "length"
        try:
            data = json.loads(content)
            if not truncated:
                return data.get("metaphors", [])
        except json.JSONDecodeError:
            pass
        print(f"[warn] 分块输出{'被截断' if truncated else ' JSON 解析失败'}（第 {attempt + 1}/{retries + 1} 次）",
              file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser(description="L4 隐喻识别（DeepSeek 版）")
    ap.add_argument("file", help="待分析文本文件")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("错误：未检测到 DEEPSEEK_API_KEY。", file=sys.stderr)
        sys.exit(1)

    text = open(args.file, encoding="utf-8").read()
    chars = count_chars(text)

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=120)

    # 分块打分
    chunks = split_text(text)
    all_metaphors = []
    failed_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  分块 {i+1}/{len(chunks)}（{len(chunk)} 字）...", file=sys.stderr)
        got = score_chunk(chunk, client, args.model)
        if got is None:
            failed_chunks.append(i)
        else:
            all_metaphors.extend(got)
    if failed_chunks:
        print(f"[warn] {len(failed_chunks)}/{len(chunks)} 个分块失败，结果偏低：{failed_chunks}", file=sys.stderr)

    # 按根域聚合
    domain_counts = {d: 0 for d in METAPHOR_DOMAINS}
    for m in all_metaphors:
        d = m.get("domain", "")
        if d in domain_counts:
            domain_counts[d] += 1

    total = len(all_metaphors)
    result = {
        "chars": chars,
        "total_metaphors": total,
        "meta_per_k": round(total / max(chars, 1) * 1000, 3),
        "domain_counts": domain_counts,
        "domain_per_k": {d: round(n / max(chars, 1) * 1000, 3) for d, n in domain_counts.items()},
        "failed_chunks": failed_chunks,
        "n_chunks": len(chunks),
        "examples": [{"text": m.get("text", "")[:60], "domain": m.get("domain", "")} for m in all_metaphors[:10]],
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"总隐喻数：{total}（每千字 {result['meta_per_k']}）")
        print("各根域密度（每千字）：")
        for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
            if n > 0:
                print(f"  {d}: {n} 处 ({result['domain_per_k'][d]}/千字)")


if __name__ == "__main__":
    main()
