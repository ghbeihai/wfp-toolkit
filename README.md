# WFP · Writing Fingerprint（文风指纹）

从一段中文文本里，自动提取**表层文风特征**的测量工具：句法、词汇、语篇衔接、隐喻根域、L3 叙事修辞软评分。

是「文笔复刻」流水线的**「表」**——负责测量"他怎么说"（数值化），不负责解释"他是谁/他怎么想"（那是 SPI / AoT 的事）。

## 测量什么

| 层 | 特征 | 例子 |
|---|---|---|
| L1 句法 | 平均句长、短句/长句占比、问句/修辞反问、我:你比例 | 平均句长 ~25 字 |
| L1 词汇 | 正/负情绪词密度、四字格密度、语气词、认知词、STTR 词汇多样性 | 语气词 ~11/千字 |
| L2 语篇 | 衔接词密度、话语标记 | 衔接词密度 |
| L3 软评分 | 17 维叙事/修辞/语域（需 LLM） | 评价/说理/引语/语域正式度 |
| L4 隐喻 | 12 根域密度（亲情/权力/旅程…） | 隐喻 ~3/千字 |

## 快速开始

```bash
# 1. 安装依赖
pip install jieba pyyaml numpy

# 2. 单文本提取硬指标（无需 LLM key）
python3 scripts/extract_hard_metrics.py 某篇文章.txt --json

# 3. 对生成文做回测（与基线比 Delta）
python3 scripts/merge_fingerprint.py 生成文指纹.json --baseline 基线.json --json

# 4. L3 软评分（需 DeepSeek key，见下）
python3 scripts/score_stylistics.py --source spi
```

## 环境变量

- `DEEPSEEK_API_KEY`：L3 软评分的 LLM key。优先读环境变量，回退到 `.env` 文件。
- 无 key 时自动进入**模拟模式**（结果带 `simulated: true`，仅用于流程验证，不可当真值）。

## 关键文件

- `scripts/extract_hard_metrics.py` — 单文本硬指标提取（L1/L2/L4，无 LLM）
- `scripts/score_stylistics.py` — L3 软评分（17 维，需 LLM）
- `scripts/merge_fingerprint.py` — 生成文 vs 基线 Delta 回测
- `config/l3_dimensions.yaml` — L3 17 维定义
- `data/thuocl_chengyu.txt` — 四字格词表（8519 条，勿删）

## 与其他模块的关系

- SPI（人格底色）+ AoT（思维模式）= **「里」**，解释作者内核；
- WFP = **「表」**，数值化作者表达；
- 三者由 `wenbi-replica` 主仓整合为人设卡，驱动风格复刻。

## License

MIT
