# ProDiPA Conditional 统一评测

本目录是后续所有 Conditional 生成方法共用的正式评价入口。配置与输出均不带版本后缀；口径调整时修改 `configs/evaluation/conditional.json`，并通过配置 SHA256 区分科研记录。

## 快速复现

从配置中已注册、已审计的方法级序列评分重建当前 canonical
Conditional 主表，可直接复制下面一行：

```bash
cd __PRODIPA_ROOT__ && bash experiments/prodipa/conditional_evaluation/run_cohort_completion_queue.sh --once
```

该命令会合并 `configs/evaluation/conditional.json` 中所有 `score_groups`，
校验 ProTrek 证据，校验并必要时重算 Novelty-U，重算整个 cohort 的
pooled HV，并完成结果强审计。`--once` 在资源或评分输入缺失时立即输出精确缺口。

同时重建 Unconditional 与 Conditional 两张当前主表：

```bash
cd __PRODIPA_ROOT__ && bash experiments/prodipa/unconditional_evaluation/run_cohort_completion_queue.sh --once && bash experiments/prodipa/conditional_evaluation/run_cohort_completion_queue.sh --once
```

## 运行前提

- Python：`__PRODIPA_ENVS__/ProDiPA/bin/python`。
- TemBERTure：`__PRODIPA_ENVS__/temberture/bin/python`。
- ProtGPT2：`__PRODIPA_MODELS__/ProtGPT2`。
- ESMFold：`__PRODIPA_MODELS__/esmfold_v1_hf`。
- ProTrek 代码：`third_party/ProTrek`。
- ProTrek 模型：`__PRODIPA_MODELS__/ProTrek_35M`。
- MMseqs2：`tools/mmseqs/bin/mmseqs`。
- UniProtKB 数据库与 `READY` 标记必须与 `configs/evaluation/conditional.json` 一致。

一行检查核心资源：

```bash
cd __PRODIPA_ROOT__ && test -s __PRODIPA_MODELS__/ProtGPT2/config.json && test -s __PRODIPA_MODELS__/esmfold_v1_hf/config.json && test -s __PRODIPA_MODELS__/ProTrek_35M/ProTrek_35M.pt && test -d third_party/ProTrek && test -s __PRODIPA_DATABASES__/uniprotkb/2026_03/READY && test -x tools/mmseqs/bin/mmseqs && nvidia-smi -L
```

当前待接入的对比方法为 `ASPO`、`CtrlProt-style`、`PrefixProt-style`、
`ProtGPT3-MSA` 和 `LineageFlow`。每种方法必须先产生 512 条方法原生的 TPR
条件生成请求，再通过本目录的同一评分入口；资源或目标家族映射缺失时不得用
替代模型或占位序列进入主表。冻结配置位于 `configs/baselines/conditional_*.json`，
资源审计入口为：

```bash
set -a
source configs/baselines/runtime_paths.env
set +a
PYTHONPATH=src python experiments/prodipa/conditional_baselines/audit_requested.py \
  --project-root "$PWD"
```

### DPLM-650M TPR motif infilling

DPLM-650M 本身不接受文本条件，因此 Conditional 实验不把“只按 TPR
长度采样”冒充为家族条件生成。`dplm_650m_tpr_infilling` 使用 DPLM
原生 `partial_masks` 机制：从三份 Conditional Train300 的 781 条唯一
TPR 序列中按 seed 42 无放回冻结 512 个请求，每个请求保留一个
34 aa 连续家族 motif，其余位点执行 500 轮掩码扩散。主表保留该明确
方法名，不与无条件 `dplm_650m` 混同。

持久队列启动命令：

```bash
cd __PRODIPA_ROOT__ && nohup env DPLM_PHYSICAL_GPU=2 DPLM_PYTHON=__PRODIPA_MODELS__/external_envs/dplm_650m/bin/python PARALLEL_GPUS=0,1,2,3 bash experiments/prodipa/conditional_baselines/run_dplm_tpr_pipeline.sh > outputs/evaluation/conditional/requested_baselines/dplm_650m_tpr_infilling/launcher.log 2>&1 < /dev/null & echo $! > outputs/evaluation/conditional/requested_baselines/dplm_650m_tpr_infilling/pipeline.pid
```

队列完成生成、统一评分、ProTrek、Novelty-U 和 pooled HV 后才原子注册
方法并重建主表。固定跟进文件为：

- `outputs/evaluation/conditional/requested_baselines/dplm_650m_tpr_infilling/pipeline.log`
- `outputs/evaluation/conditional/requested_baselines/dplm_650m_tpr_infilling/logs/generation.log`
- `outputs/evaluation/conditional/requested_baselines/dplm_650m_tpr_infilling/logs/protrek.log`
- `outputs/evaluation/conditional/requested_baselines/dplm_650m_tpr_infilling/STATUS.json`
- `outputs/evaluation/conditional/methods/dplm_650m_tpr_infilling/scoring_queue.log`

## 冻结条件与输入

每个方法提交一份 TSV，必需列为：

`paradigm, model, sample_id, seed, sequence, length_aa, valid_sequence, eos_reached, truncated, valid_complete, wrapper_closed, invalid_reason, raw_completion, temperature, top_p, top_k, repetition_penalty, max_new_tokens`

- 指令：`[Generate by superfamily]`
- 条件：`Superfamily=<Tetratricopeptide-like helical domain superfamily>`
- 目标条件：TPR 超家族；不再执行单独的 Family Match 评分。
- 每个方法提交 512 次请求，失败请求同样保留一行。
- `paradigm` 固定为 `conditional`；`model` 必须与发布时的 `model-id` 一致。
- `sample_id` 在方法内唯一；不得删除失败请求或用额外生成进行补齐。
- 标准有效序列只接受 20 种标准氨基酸，长度 50–512 aa。
- ProLLaMA 输出单独审计 `Seq=<...>` wrapper 是否闭合；不使用该 wrapper 的方法按其原生完整性标记处理。所有方法仍必须明确记录 EOS/停止状态和截断状态。

参考自回归协议固定为 seed 42、temperature 0.7、top-p 0.9、top-k 200、repetition penalty 1.2、最多 512 个新 token。机制不同的方法可使用其原生解码，但必须保持相同 TPR 条件、请求预算并逐请求记录真实参数。

## 正式指标

Canonical 主表固定为：PPL、ProTrek %、pLDDT、pAE、Sol、Thermo、HV。

- PPL：冻结原始 ProtGPT2。
- ProTrek %：冻结 TPR 文本条件与生成序列在共享嵌入空间中的余弦相似度。
- pLDDT/pAE：冻结 ESMFold。
- Sol：冻结 Protein-Sol。
- Thermo：冻结 TemBERTureCLS。
- HV：`-log(1+PPL)`、ProTrek %、pLDDT、Sol、Thermo 五个序列级目标，在完整 Conditional cohort 上统一 min-max，参考点为零向量。

Novelty-U、Novelty-T、Valid/EOS Rate、严格 JSR 和长度分布保留为补充分析。严格 JSR 阈值为 pLDDT ≥85、Sol ≥0.7、Thermo ≥0.6，分母为全部请求。

ProTrek 是 canonical 发布的必需条件相似度证据。基础代理评分可以先断点完成，但缺少该证据时只能标记为等待，不能形成可引用的完整主表。

## 评价并发布一个新方法

```bash
cd __PRODIPA_ROOT__

bash experiments/prodipa/conditional_evaluation/run_method_evaluation.sh \
  --model-id descriptive_method_name \
  --generation-tsv /absolute/path/descriptive_method_name.tsv \
  --protrek-tsv /absolute/path/protrek.tsv \
  --gpus 0,1,2,3 \
  --publish
```

该入口依次执行：

1. Fail-closed 校验 512 条请求、条件 wrapper、EOS、有效字符与长度。
2. 保存生成文件 SHA256 和完整评分协议指纹；输入或口径变化时禁止复用旧结果。
3. 运行 PPL、ESMFold、Protein-Sol、TemBERTure 和共同 Novelty-T。
4. 校验并合并 ProTrek 条件相似度证据。
5. 原子注册完成方法，合并完整 cohort，统一计算 Novelty-U 与 pooled HV。
6. 通过强审计后发布 canonical 主表。

如果暂时没有 ProTrek 文件，可省略对应参数并先完成基础评分；同时省略 `--publish`。补齐证据后使用相同命令和输出目录恢复，已完成评分不会重复执行。

### 从生成 TSV 到主表的单行命令

下面命令先完成基础评分，再对实际有效唯一序列计算 ProTrek，最后断点恢复、
注册方法并重算主表。只需替换 `MODEL_ID` 和 `GENERATION_TSV`：

```bash
cd __PRODIPA_ROOT__ && MODEL_ID=descriptive_method_name GENERATION_TSV=/absolute/path/to/generation.tsv OUT=outputs/evaluation/conditional/methods/descriptive_method_name bash -c 'bash experiments/prodipa/conditional_evaluation/run_method_evaluation.sh --model-id "$MODEL_ID" --generation-tsv "$GENERATION_TSV" --gpus 0,1,2,3 && CUDA_VISIBLE_DEVICES=0 __PRODIPA_ENVS__/ProDiPA/bin/python experiments/prodipa/conditional_evaluation/score_protrek_similarity.py --input "$OUT/prepared/unique_valid_sequences.tsv" --output "$OUT/scores/protrek.generated.tsv" --repo third_party/ProTrek --model __PRODIPA_MODELS__/ProTrek_35M --device cuda:0 && bash experiments/prodipa/conditional_evaluation/run_method_evaluation.sh --model-id "$MODEL_ID" --generation-tsv "$GENERATION_TSV" --protrek-tsv "$OUT/scores/protrek.generated.tsv" --gpus 0,1,2,3 --publish'
```

如已有与该生成 TSV 完全对应的 ProTrek TSV，则直接使用更短的一行命令：

```bash
cd __PRODIPA_ROOT__ && bash experiments/prodipa/conditional_evaluation/run_method_evaluation.sh --model-id descriptive_method_name --generation-tsv /absolute/path/to/generation.tsv --protrek-tsv /absolute/path/to/protrek.tsv --gpus 0,1,2,3 --publish
```

## 完整执行顺序

1. `validate_generation.py` 校验 512 条请求、冻结条件、wrapper、EOS、字符集和长度。
2. 生成 TSV 被固定到方法的 `input/generation/`，并记录生成文件和评分协议 SHA256。
3. CPU 任务并行计算 Protein-Sol 与共同 Novelty-T。
4. GPU 队列计算 ProtGPT2 PPL、TemBERTure 和 16 个 ESMFold 分片。
5. 合并 ESMFold 分片，产生基础序列级评分。
6. `score_protrek_similarity.py` 使用冻结条件文本计算有效唯一序列的 ProTrek %。
7. 验证 ProTrek `sequence_id` 与方法有效序列一一对应，再写入方法级结果。
8. `--publish` 注册方法，合并完整 cohort，计算 Novelty-U 和 pooled HV，最后执行强审计。

同一 `model-id` 的生成文件或评分协议改变时会 fail-closed。需要对不同生成进行独立评测时，
应使用新的可读 `model-id`，不覆盖已审计输入。

## Base 与三个教师

当前 Base、Fold Teacher、Sol Teacher 和 Thermo Teacher 使用同一统一根目录：

`outputs/evaluation/conditional/`

四组 512 次生成及基础评分完成后，该目录就是首个 Conditional score group。将完整条件相似度评分放在 `scores/protrek.tsv` 后执行：

```bash
bash experiments/prodipa/conditional_evaluation/finalize_current_group.sh --once
```

该入口先验证条件相似度表与全部有效唯一序列一一对应，重建 Base/三教师汇总，再进入 cohort 完成队列。`--once` 只检查一次剩余依赖并给出精确缺口；移除后会等待冻结 UniProtKB 数据库就绪并自动完成统一汇总。

## 正式产物

- `outputs/evaluation/conditional/results/conditional_prodipa_main.tsv`
- `outputs/evaluation/conditional/results/conditional_sequence_scores.tsv`
- `outputs/evaluation/conditional/results/conditional_hv_manifest.json`
- `outputs/evaluation/conditional/results/conditional_supplementary_metrics.tsv`
- `outputs/evaluation/conditional/results/conditional_scoring_completeness.tsv`
- `outputs/evaluation/conditional/EVALUATION_PROTOCOL.json`
- `outputs/evaluation/conditional/RESULT_AUDIT.json`
- `outputs/evaluation/conditional/COMPLETE.json`

只有 `RESULT_AUDIT.json` 为 `PASS` 且 `STATUS.json` 为 `COMPLETE` 时，主表才是可引用的 canonical 结果。新增方法后必须在整个 cohort 上重新归一化并重算 HV，旧 HV 不得直接沿用。

一行检查最终发布状态：

```bash
cd __PRODIPA_ROOT__ && __PRODIPA_ENVS__/ProDiPA/bin/python -c 'import json; from pathlib import Path; r=Path("outputs/evaluation/conditional"); print(json.loads((r/"STATUS.json").read_text())); print(json.loads((r/"RESULT_AUDIT.json").read_text()))'
```
