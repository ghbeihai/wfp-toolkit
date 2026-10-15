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
| L4 隐喻 | 12 根域密度（旅程/战争/建筑/生命…，DeepSeek 语境识别） | 隐喻 ~3/千字 |

## 快速开始

```bash
# 1. 安装依赖（推荐用 wenbi-replica/install.sh 一次装齐）
pip install jieba pyyaml numpy openai python-dotenv

# 2. 单文本提取硬指标（无需 LLM key；空文本报错退出 1）
python3 scripts/extract_hard_metrics.py 某篇文章.txt --json

# 3. 逐篇提取 + 聚合建基线（<5 篇报错；--min-docs 可调；doc_ids 写入基线）
for f in 原文/*.txt; do python3 scripts/extract_hard_metrics.py "$f" --json > fp/$(basename "$f" .txt).json; done
python3 scripts/extract_hard_metrics.py --summary fp/*.json --name 作者名 > fp/wfp_baseline_作者名.json

# 4. 判别力自检（两关：对照文 ≥1.5、异作者 ≥1.2；--strict 时 FAIL 退出 2）
python3 scripts/validate_discrimination.py --baseline fp/wfp_baseline_作者名.json \
    --held-out 留出.txt --control 对照.txt --other-author 他人.txt
python3 scripts/validate_discrimination.py --baseline fp/wfp_baseline_作者名.json \
    --corpus ./原文 --other-author ./他人语料 --rounds 5 --seed 42

# 5. 对生成文做回测（与基线比 Delta）
python3 scripts/merge_fingerprint.py 生成文指纹.json --baseline 基线.json --json

# 6. L3 软评分（需 DeepSeek key）：先 --dry-run 看调用次数
python3 scripts/score_stylistics.py --corpus-dir ./原文 --dry-run
python3 scripts/score_stylistics.py --corpus-dir ./原文 --out-dir ./l3_scores --l3-baseline-out l3_baseline.json
python3 scripts/score_metaphor.py 某篇文章.txt --out metaphor.json   # L4 隐喻（独立输出，暂不进 Delta）
```

## Delta 口径（field_map v4）

- 所有 Delta 消费者（validate / merge / normalize / replicate / iterative）共用 `scripts/field_map.py`；
- `L1_LEX_002_ttr_word`（原始 TTR，随篇长单调下降）不进 Delta，词汇多样性用 STTR；
- 我:你、正:负 比值用 +1 平滑（(a+1)/(b+1)），分母为 0 不再爆炸；旧版基线自动跳过这两项；
- 6 个句长相关特征共享 3.0 的权重（每个 0.5），避免同一信号重复计票；
- 基线 <5 篇或可用特征 <10 项直接报错（DegenerateBaseline），不再输出 999/判别比 1.0；sd 为样本标准差（n-1）。
- **v4 变更**：篇内句长 sd（`L1_SYN_002_sd_sl`）与 burstiness（`L1_SYN_007_burstiness`）由总体标准差改为样本标准差（n-1），与基线口径一致。
  数值变化：单篇 sd 乘以 √(n/(n-1))（n=句数；30 句的文章约 +1.7%，10 句约 +5.4%）；burstiness = (sd−μ)/(sd+μ) 随之略升。
  v3 及更早的基线自动跳过这两项（`CHANGED_IN`），请重建基线。张林军自检（同一 343 篇训练集重建 v4 基线）：判别比均值 2.13 / 2.69 与 v3 相同，通过数 10/12、12/12 不变。

## 多尺子综合分（ruler.py）

- 分层子尺：L1 句法 / L1 词汇 / L2 语篇 / L3 软评分（可选）/ L4 隐喻与价值（含 LLM 隐喻密度 `L4_LLM_meta_per_k`）。
- `ruler.py extend --baseline B --metaphor-dir M --out B2`：把 score_metaphor 结果并入基线（extra 节，≥5 篇）——隐喻进 Delta。
- `ruler.py learn --baseline B --pos 原作者留出指纹目录 --neg 对照/异作者指纹目录 --length-fps 基线指纹目录 --out w.json`：
  每个特征权重 = 先验 × max(2·AUC−1, 0.05)；与篇长 |ρ|≥0.4 的再 ×0.5。**只降权不删特征**；总权重归一到先验总和，综合分与旧 Delta 同量级。
- `ruler.py score fp.json --baseline B --weights w.json`：综合分 + 分层分；`validate_discrimination.py --weights w.json` 用学习权重做两关自检。
- `blind_test.py --reference 原文目录 --real 真文 --fake 模仿文`：LLM 盲测尺（正反顺序各一次），P(识破) 越接近 0.5 越以假乱真；独立信号，不并入综合分。
- `genre.py`：体裁识别（讲稿/随笔/新闻/对话/诗歌/其他；规则 + 可选 `--llm`）与语料闸门。

## 缓存

- 单篇指纹：按 (文本 + 提取口径版本) 内容哈希缓存；LLM 调用：按请求参数哈希缓存（只缓存成功结果）。
- 目录 `$WENBI_CACHE_DIR`（默认 `~/.cache/wenbi`）；`WENBI_NO_CACHE=1` 或 `extract_hard_metrics.py --no-cache` 关闭。

## 环境变量与配置

- `DEEPSEEK_API_KEY`：L3 软评分 + L4 隐喻识别需要。查找顺序：环境变量 → `$WENBI_ENV_FILE` → 本仓 `.env` → `$AOT_ROOT/$SPI_ROOT/.env`；**没 key 直接报错退出 1**。
- `config/llm.yaml`：model / temperatures / 并发 / 重试与退避 / 截断阈值（max_text_chars，截断会在结果里标 truncated）。
- `config/data_sources.yaml`：`${SPI_ROOT}`、`${AOT_ROOT}`、`${WFP_OBJ}` 从环境变量解析，未解析时报错；也可直接用 `--spi-path` / `--aot-dir` 覆盖。
- `WENBI_MOCK_LLM=1`（或 `flaky`）：用本地假客户端跑通流程（测试用，不花钱、不需要 key）。

## 关键文件

- `scripts/extract_hard_metrics.py` — 硬指标提取 + 批量建基线
- `scripts/field_map.py` — 指纹扁平化 / 基线聚合 / Delta 唯一口径
- `scripts/validate_discrimination.py` — 判别力两关自检（`--weights` 用学习权重）
- `scripts/ruler.py` — 多尺子分层打分与权重学习
- `scripts/genre.py` — 体裁识别与语料闸门
- `scripts/blind_test.py` — LLM 盲测尺
- `scripts/score_stylistics.py` / `build_l3_baseline.py` — L3 软评分与基线
- `scripts/score_metaphor.py` — L4 隐喻识别
- `scripts/merge_fingerprint.py` / `normalize_backtest.py` — 生成文 vs 基线 Delta 回测
- `scripts/llm_common.py` / `data_sources.py` — LLM 与数据源公共模块
- `config/l3_dimensions.yaml` — L3 17 维定义
- `data/thuocl_chengyu.txt` — 四字格词表（8519 条，勿删）

## 与其他模块的关系

- SPI（人格底色）+ AoT（思维模式）= **「里」**，解释作者内核；
- WFP = **「表」**，数值化作者表达；
- 三者由 `wenbi-replica` 主仓整合为人设卡，驱动风格复刻。

## License

MIT
