# ProDiPA Unconditional 统一评测

本目录是后续所有 Unconditional 生成方法共用的正式评价入口。目录名、配置名和输出名不带版本后缀；评价口径需要调整时，直接修改本目录及 `configs/evaluation/unconditional.json`，并在科研记录中保留配置哈希。

## 快速复现

从配置中已注册、已审计的方法级序列评分重建当前 canonical
Unconditional 主表，可直接复制下面一行：

```bash
cd __PRODIPA_ROOT__ && bash experiments/prodipa/unconditional_evaluation/run_cohort_completion_queue.sh --once
```

该命令会合并 `configs/evaluation/unconditional.json` 中的所有
`score_groups`，校验并必要时重算 Novelty-U，重算整个 cohort 的
pooled HV，最后执行强审计。`--once` 表示如有输入缺失则立即报出精确路径，
不在终端中无限等待。这是从已完成方法评分重建主表，不重复加载 GPU
评分器。

对一份新的 512 请求生成 TSV 执行完整评分、注册和主表发布，可复制这一行，
只替换 `MODEL_ID` 和 `GENERATION_TSV`：

```bash
cd __PRODIPA_ROOT__ && MODEL_ID=descriptive_method_name GENERATION_TSV=/absolute/path/to/generation.tsv bash -c 'bash experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh --model-id "$MODEL_ID" --generation-tsv "$GENERATION_TSV" --gpus 0,1,2,3 --publish'
```

## 运行前提

- Python：`__PRODIPA_ENVS__/ProDiPA/bin/python`。
- TemBERTure：`__PRODIPA_ENVS__/temberture/bin/python`。
- ProtGPT2：`__PRODIPA_MODELS__/ProtGPT2`。
- ESMFold：`__PRODIPA_MODELS__/esmfold_v1_hf`。
- MMseqs2：`tools/mmseqs/bin/mmseqs`。
- UniProtKB 库与就绪标记必须与 `configs/evaluation/unconditional.json`
  中的 `novelty_u_database` 和 `novelty_u_ready_marker` 一致。
- GPU 编号是 `nvidia-smi` 的物理编号；队列只在显存条件满足时加载模型。

可用下面命令快速检查核心资源：

```bash
cd __PRODIPA_ROOT__ && test -s __PRODIPA_MODELS__/ProtGPT2/config.json && test -s __PRODIPA_MODELS__/esmfold_v1_hf/config.json && test -s __PRODIPA_DATABASES__/uniprotkb/2026_03/READY && test -x tools/mmseqs/bin/mmseqs && nvidia-smi -L
```

## 固定输入协议

每个方法提交一份 TSV，共 512 次生成请求。必需列为：

`paradigm, model, sample_id, seed, sequence, length_aa, valid_sequence, eos_reached, truncated, valid_complete, wrapper_closed, invalid_reason, raw_completion, temperature, top_p, top_k, repetition_penalty, max_new_tokens`

- `paradigm` 固定为 `unconditional`。
- `model` 使用可读的方法名，不使用 B2、P1 等模糊缩写。
- `sample_id` 在模型内唯一；无效请求也必须保留一行。
- 参考自回归协议为 seed 42、temperature 0.7、top-p 0.9、top-k 200、repetition penalty 1.2、最多 512 个新 token。DPLM 等机制不同的外部方法可以采用其官方 method-native 解码，但必须逐请求记录实际参数；不得把不适用的自回归参数伪装成已使用。
- 有效序列仅接受 20 种标准氨基酸，长度为 50–512 aa。
- 失败请求不得删除，也不得用其他样本补齐。

## 固定指标

主表固定为：PPL、Novelty-U、Novelty-T、pLDDT、%>70、pAE、%<10、Sol、Thermo、HV。

- PPL：冻结原始 ProtGPT2。
- Novelty-U：MMseqs2 对 UniProtKB 2026_03 canonical Swiss-Prot + TrEMBL 的最近邻不相似度。
- Novelty-T：所有方法共同使用 Fold/Sol/Thermo 三份 300 条训练集的并集。
- pLDDT/pAE：冻结 ESMFold。
- Sol：冻结 Protein-Sol。
- Thermo：冻结 TemBERTureCLS。
- 连续均值只在有效且成功评分的序列上计算，同时保留完整请求分母的 Valid Rate、EOS Rate、JSR 和评分完整度作为补充表。

HV 严格使用五个序列级目标：`-log(1+PPL)`、Novelty-U、pLDDT、Sol、Thermo。所有对比方法的完整序列级点合并后统一 min-max 到 `[0,1]`，参考点为零向量，再对每个方法的非支配集合计算精确 HV。Novelty-T、pAE、%>70 和 %<10 不进入 HV。新增方法后必须对整个 cohort 重算 HV，旧 HV 不能直接沿用。

## 评价并发布一个新方法（推荐）

```bash
cd __PRODIPA_ROOT__

bash experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh \
  --model-id descriptive_method_name \
  --generation-tsv /absolute/path/to/descriptive_method_name.tsv \
  --gpus 0,1,2,3 \
  --publish
```

这是后续新方法的单一正式入口。它依次执行：

1. 从 canonical 配置读取请求数、有效长度、严格 JSR 阈值和 ESMFold 分片数。
2. fail-closed 校验 512 条请求，保存源文件 SHA256；同一 `model-id` 不允许静默换数据。
3. 保存评分协议指纹；评分开始后若冻结模型、阈值、资源参数或参考数据库发生变化，旧 `.done` 结果不得静默复用。
4. 执行 ProtGPT2 PPL、ESMFold、Protein-Sol、TemBERTure 和共同 Novelty-T。
5. 评分全部完成后，在文件锁保护下原子注册到 `configs/evaluation/unconditional.json`。
6. 合并完整 cohort；若查询集合、UniProtKB 引用和 MMseqs2 版本均未改变则复用 Novelty-U，否则重新计算。
7. 在完整 cohort 上重新归一化并计算 pooled HV，通过结果强审计后发布 canonical 主表。

该入口支持 `.done` 级断点续跑，并沿用共享 GPU 检测与项目 GPU UUID 锁。等待 GPU 时不加载模型。仅想完成单方法评分而暂不加入正式 cohort 时，省略 `--publish`。

## 完整执行顺序

1. `validate_generation.py` 校验 512 条请求、方法名、序列字符、长度和完整性字段。
2. 生成 TSV 被复制到方法的 `input/generation/`，并记录源文件和评分协议 SHA256。
3. CPU 任务并行执行 Protein-Sol 与共同 Novelty-T。
4. GPU 队列执行 ProtGPT2 PPL、TemBERTure 和 16 个 ESMFold 分片。
5. 合并 ESMFold 分片，生成方法级 `sequence_scores.tsv`、补充指标和完整度表。
6. `--publish` 原子注册方法，合并全 cohort，计算 Novelty-U 和 pooled HV。
7. `audit_results.py` 检查模型集合、请求数、字段范围、HV manifest 与结果完整性。

评分队列是断点可恢复的。同一 `model-id` 的生成文件或冻结协议发生变化时会
fail-closed，不会静默复用旧评分。

单方法状态与数据证据固定保存在：

- `outputs/evaluation/unconditional/methods/<model-id>/input/generation_validation.json`
- `outputs/evaluation/unconditional/methods/<model-id>/input/generation.sha256`
- `outputs/evaluation/unconditional/methods/<model-id>/SCORING_STATUS.json`
- `outputs/evaluation/unconditional/methods/<model-id>/METHOD_EVALUATION_COMPLETE.json`
- `outputs/evaluation/unconditional/methods/<model-id>/results/sequence_scores.tsv`

如果单方法此前未使用 `--publish`，可在确认后注册并重建 cohort：

```bash
PY=__PRODIPA_ENVS__/ProDiPA/bin/python

$PY experiments/prodipa/unconditional_evaluation/register_method.py \
  --project-root __PRODIPA_ROOT__ \
  --config configs/evaluation/unconditional.json \
  --model-id descriptive_method_name \
  --evaluation-root outputs/evaluation/unconditional/methods/descriptive_method_name

bash experiments/prodipa/unconditional_evaluation/run_cohort_completion_queue.sh
```

注册操作是幂等的；同一方法指向不同结果目录会 fail-closed。cohort 队列等待所有方法评分及冻结 UniProtKB MMseqs2 数据库就绪，计算 Novelty-U，并重新生成 pooled HV。正式输出固定在：

- `outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv`
- `outputs/evaluation/unconditional/results/unconditional_sequence_scores.tsv`
- `outputs/evaluation/unconditional/results/unconditional_hv_manifest.json`
- `outputs/evaluation/unconditional/results/unconditional_supplementary_metrics.tsv`
- `outputs/evaluation/unconditional/results/unconditional_scoring_completeness.tsv`
- `outputs/evaluation/unconditional/COMPLETE.json`
- `outputs/evaluation/unconditional/EVALUATION_PROTOCOL.json`
- `outputs/evaluation/unconditional/RESULT_AUDIT.json`

只有 `RESULT_AUDIT.json` 为 `PASS` 且 `STATUS.json` 为 `COMPLETE` 时，`outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv` 才是可引用的 canonical 主表。单方法目录里的同名表是 pooled HV 之前的阶段性表，不应用于跨方法最终排名。

一行检查最终发布状态：

```bash
cd __PRODIPA_ROOT__ && __PRODIPA_ENVS__/ProDiPA/bin/python -c 'import json; from pathlib import Path; r=Path("outputs/evaluation/unconditional"); print(json.loads((r/"STATUS.json").read_text())); print(json.loads((r/"RESULT_AUDIT.json").read_text()))'
```

## 固定补充阈值与资源上限

- Strict JSR：pLDDT ≥85、Sol ≥0.7、Thermo ≥0.6，以全部请求为分母。
- `%>70` 与 `%<10` 是冻结主表指标，不随 Strict JSR 阈值改变。
- ESMFold 默认 16 分片。
- UniProtKB Novelty-U 默认 32 CPU 线程，并通过 `split-memory-limit=300G` 限制 MMseqs2 峰值内存。
- Novelty-U 缓存必须同时匹配完整查询 `sequence_id` 集合、UniProtKB 标签和数据库绝对路径；否则不得复用。

这些值全部以 `configs/evaluation/unconditional.json` 为唯一配置源。修改配置会改变配置 SHA256，并要求重新发布完整 cohort。

不得把不同 UniProtKB 版本、不同 Novelty-T 训练参考、不同请求预算或不同解码协议的结果直接追加到主表。需要改变上述任一条件时，应重评完整 cohort，并在结果报告中明确差异。

## 待接入的五个正式 baseline

ESM2、ESM3、ProteinDT、Pinal 和 ASPO 已建立独立资源审计与后置执行队列。它们在完成 512 次真实生成、统一评分及结果审计以前，不写入 canonical `score_groups`，也不把外部论文中的汇总均值当成本项目结果。

当前资源与执行状态汇总见 `reports/UNCONDITIONAL_ADDITIONAL_BASELINES_STATUS.md`。

固定实现口径：

- `esm2_650m`、`esm3_open_1.4b`：按照冻结的掩码填充协议，从 512 条自然参考序列出发，每轮无放回掩码 10% 位点，直到所有位点完成一次填充。
- `proteindt_t5`、`pinal_1.2b`：固定提示 `Design novel protein sequences with high foldability, high solubility, and high thermostability.`。
- `aspo_esm2_650m`：Fold、Sol、Thermo steering vector 权重固定为 `0.5 / 0.5 / 1.0`；方法名明确保留 ESM2-650M backbone，避免把不同 ASPO 实例混为一谈。
- 五个方法都使用 512 请求、seed 42；生成完成后调用本目录的统一评测入口，并在完整 cohort 上重新计算 Novelty-U 与 pooled HV。

冻结参考序列及来源证据：

- `data/evaluation/unconditional/masked_infilling_reference_sequences.tsv`
- `data/evaluation/unconditional/masked_infilling_reference_manifest.json`

资源路径填写在 `configs/baselines/runtime_paths.env`，变量模板见 `configs/baselines/runtime_paths.env.example`。启动持久队列：

```bash
nohup bash experiments/prodipa/unconditional_evaluation/run_requested_baselines_queue.sh \
  > outputs/evaluation/unconditional/requested_baselines/launcher.log 2>&1 < /dev/null &
echo $! > outputs/evaluation/unconditional/requested_baselines/queue.pid
```

队列在资源不完整时只轮询文件系统，不加载 CUDA 模型。机器可读状态为：

- `outputs/evaluation/unconditional/requested_baselines/STATUS.json`
- `outputs/evaluation/unconditional/requested_baselines/resource_audit.json`
- `outputs/evaluation/unconditional/requested_baselines/preflight/<method>.json`
- `outputs/evaluation/unconditional/requested_baselines/queue.log`
