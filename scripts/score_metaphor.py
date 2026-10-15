#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4 隐喻识别（DeepSeek 版）—— 替代原词典匹配的隐喻密度统计。

为什么用 LLM：隐喻识别是 NLP 难题，词典只能抓"旅程/战争"这类字面词，
语境隐喻（"他的事业触礁了""人生是一场马拉松"）词典抓不到，LLM 能识别。

输入：一篇中文文本
输出：JSON —— 总隐喻数、12 根域各自密度（每千字）、例句列表

用法：
  python3 score_metaphor.py 某篇文章.txt --json
  DEEPSEEK_API_KEY=sk-... python3 score_metaphor.py file.txt --json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# 12 隐喻根域（冻结自 AoT 尺子维度6，与原词典版对齐）
METAPHOR_DOMAINS = {
    "旅程": "把人生/事业/关系比作旅程：路、方向、终点、前行、迷路、跨越、脚步",
    "战争": "把争论/竞争/对抗比作战争：战斗、敌人、进攻、防守、胜利、攻克、武器",
    "建筑": "把体系/人格/作品比作建筑：地基、框架、支柱、搭建、倒塌、支撑、加固",
    "机器": "把组织/身体/社会比作机器：运转、零件、齿轮、引擎、卡住、机制、启动",
    "生命": "把抽象事物比作生物：成长、种子、根、开花、结果、养分、发芽、枯萎",
    "市场": "把关系/能力/事物比作商品交易：价值、成本、投资、回报、交易、利润、筹码",
    "游戏": "把生活/竞争比作游戏博弈：规则、对手、一局、赌注、输赢、出牌、玩家",
    "剧场": "把社会/人生比作舞台戏剧：舞台、登场、扮演、角色、剧本、谢幕、观众",
    "容器": "把抽象状态比作容器：里面、外面、充满、溢出、容纳、装进、掏空、塞满",
    "光暗": "把情绪/认知比作光暗：光明、黑暗、照亮、阴影、阳光、乌云、曙光、阴霾",
    "上下": "把地位/数量/情绪比作垂直方位：上升、下降、提高、跌落、顶层、底层、攀坠",
    "温度": "把关系/情绪比作温度：热、冷、温暖、冰冷、热情、冷淡、寒意、升温、降温",
}

SYSTEM_PROMPT = """你是一位严谨的隐喻识别编码员。

你会收到一段中文文本。任务：通读全文，找出所有【概念隐喻】——即把抽象事物比作具体事物的表达（不是明喻"像...一样"就算，是"用一个域的词描述另一个域"才算）。

例如：
- "他的事业触礁了" → 事业比作船/旅程（旅程域）
- "这件事在我心里埋下了种子" → 想法比作植物（生命域）
- "他的观点站不住脚" → 观点比作建筑（建筑域）
- "我们之间的关系降温了" → 关系比作温度（温度域）

输出严格 JSON，无其他文字：
{
  "metaphors": [
    {"text": "原文片段（逐字引用）", "domain": "12根域之一", "explain": "为什么是这个域，20字内"}
  ]
}

12 个根域只能是：旅程、战争、建筑、机器、生命、市场、游戏、剧场、容器、光暗、上下、温度。
找不到任何隐喻时输出 {"metaphors": []}。
只识别真隐喻，不要把普通修辞、成语、字面用法算成隐喻。"""


def load_key() -> str | None:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    # 回退到 .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def count_chars(text: str) -> int:
    """中文字符数（用于每千字密度）。"""
    return len(re.sub(r"\s+", "", text))


def score_metaphor(text: str, client, model: str = "deepseek-chat") -> dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = resp.choices[0].message.content.strip()
    # 容错：去掉 markdown 代码块包裹
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    return data.get("metaphors", [])


def main():
    ap = argparse.ArgumentParser(description="L4 隐喻识别（DeepSeek 版）")
    ap.add_argument("file", help="待分析文本文件")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("错误：未检测到 DEEPSEEK_API_KEY。\n"
              "请先 export DEEPSEEK_API_KEY=sk-... 或在 .env 文件中配置。",
              file=sys.stderr)
        sys.exit(1)

    text = open(args.file, encoding="utf-8").read()
    chars = count_chars(text)

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=120)

    metaphors = score_metaphor(text, client, args.model)

    # 按根域聚合
    domain_counts = {d: 0 for d in METAPHOR_DOMAINS}
    for m in metaphors:
        d = m.get("domain", "")
        if d in domain_counts:
            domain_counts[d] += 1

    total = len(metaphors)
    result = {
        "chars": chars,
        "total_metaphors": total,
        "meta_per_k": round(total / max(chars, 1) * 1000, 3),
        "domain_counts": domain_counts,
        "domain_per_k": {d: round(n / max(chars, 1) * 1000, 3) for d, n in domain_counts.items()},
        "examples": [{"text": m.get("text", "")[:60], "domain": m.get("domain", "")} for m in metaphors[:10]],
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
