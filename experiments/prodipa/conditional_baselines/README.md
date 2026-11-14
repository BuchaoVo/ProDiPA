# Conditional comparison methods

本目录管理五个 TPR 条件生成对比。正式比较固定为 512 次请求、seed 42、
目标家族 `G3DSA:1.25.40.10`，生成后统一进入
`experiments/prodipa/conditional_evaluation/`。

输入准备不会重新计算属性分数：

```bash
python experiments/prodipa/conditional_baselines/prepare_inputs.py \
  --project-root "$PWD"
```

资源审计：

```bash
set -a; source configs/baselines/runtime_paths.env; set +a
PYTHONPATH=src python experiments/prodipa/conditional_baselines/audit_requested.py \
  --project-root "$PWD"
```

后台队列先等待当前 Conditional 评分完成；资源未就绪时不加载模型、不占显存。
每个方法生成后执行统一属性评分和 ProTrek 条件相似度，再通过注册审计进入主表。
Family Match 与 Family-JSR 不在当前协议中。

```bash
nohup bash experiments/prodipa/conditional_baselines/run_requested_queue.sh \
  > outputs/evaluation/conditional/requested_baselines/queue.log 2>&1 < /dev/null &
echo $! > outputs/evaluation/conditional/requested_baselines/queue.pid
```

冻结 ProLLaMA 激活向量使用单独资源队列准备。公开实现依赖未随仓库发布的
`self_attn.head_out`，本项目明确映射到标准 Hugging Face LLaMA 的
`self_attn.o_proj`，并以 `aspo_tpr_hf_compat` 标识，不能写成逐字复现：

```bash
nohup python experiments/prodipa/conditional_baselines/run_aspo_resource_queue.py \
  --project-root "$PWD" --allowed-gpus 2,3 \
  > outputs/evaluation/conditional/requested_baselines/aspo_resource_queue.out 2>&1 < /dev/null &
echo $! > outputs/evaluation/conditional/requested_baselines/aspo_resource_queue.pid
```

运行状态与日志：

- `outputs/evaluation/conditional/requested_baselines/STATUS.json`
- `outputs/evaluation/conditional/requested_baselines/RESOURCE_AUDIT.json`
- `outputs/evaluation/conditional/requested_baselines/ASPO_RESOURCE_STATUS.json`
- `outputs/evaluation/conditional/requested_baselines/aspo_resource_queue.log`
- `outputs/evaluation/conditional/requested_baselines/protgpt3_msa_download.log`
- `outputs/evaluation/conditional/requested_baselines/logs/<method>.log`

LineageFlow 只在 CATH 目标家族与其已发布 Pfam/ASR 资产存在可核验证据映射时
运行。仅凭“Tetratricopeptide repeat”名称相似不能视为同一评价条件。
