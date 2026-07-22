# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

本文件与 `CLAUDE.md` 独立维护；修改共享的项目事实、命令或参数时，必须同步核对两份文件。

## 项目概述

MiniMind 是一个从零实现的小型语言模型训练框架，内置完整训练管线：预训练 → SFT → DPO/RLHF → 推理模型训练。模型主体使用 PyTorch 实现，不使用 transformers 的内置模型架构；项目复用其配置/模型基类、GenerationMixin、标准输出类型和 tokenizer 接口。

## 项目结构

```
model/
  MiniMindModel.py   — 模型架构（Config、RMSNorm、RoPE+YaRN、Attention、SwiGLU FFN、MoE、Transformer Block、ForCausalLM）
  model_lora.py       — LoRA 适配器（apply_lora / load_lora / save_lora）
  tokenizer.json       — ChatML 分词器（tokenizer_config.json 含完整 chat_template，支持 tool calling）
trainer/
  train_pretrain.py    — 预训练（Next-Token Prediction）
  train_full_sft.py    — 全参数有监督微调
  train_lora.py        — LoRA 微调（只训练 QKV+O 投影的低秩适配器）
  train_dpo.py         — Direct Preference Optimization
  train_ppo.py         — PPO 强化学习（Actor-Critic + Reward Model）
  train_grpo.py        — Group Relative Policy Optimization（去掉了 Critic 的简化 RL）
  train_reason.py      — 推理模型蒸馏训练（学习 DeepSeek-R1 风格的 <think>...<answer> 格式）
  trainer_utils.py     — 共享工具（init_model、lm_checkpoint、SkipBatchSampler、分布式初始化、动态学习率等）
dataset/
  llm_dataset.py       — 4 个 Dataset 类：PretrainDataset / SFTDataset / DPODataset / RLAIFDataset
data_registry/
  smoke.yaml            — smoke 数据集注册表
eval/
  smoke_test.py         — 框架正确性快速测试（7 个训练管线，50 步）
  benchmark.py          — 模型质量深度评估（PPL / 生成质量 / Reasoning 格式）
  eval_utils.py         — 共享工具（报告生成、SwanLab 集成、梯度检查）
  test_data/            — smoke test 用小型 JSONL 数据集（5 类）
  reports/              — 评测报告输出目录（gitignored）
evals/
  README.md             — 评估系统使用说明
  run_all.py            — 一键运行全部评估 + 汇总报告
  eval_lm.py            — 语言模型评估（PPL / Validation Loss）
  eval_qa.py            — 领域问答评估（Exact Match / Keyword Recall）
  eval_generation.py    — 约束生成评估（JSON / 长度 / 关键词）
  eval_speed.py         — 推理性能评估（tokens/s / 延迟 / 显存）
  compare_runs.py       — 两次评估结果对比
  core/                 — 核心模块（load_model / metrics / io_utils / report）
  configs/              — YAML 配置文件
  data/                 — 样例评估数据
feedback/              — 评测失败分析与人工审核 SFT 候选生成
scripts/
  audit_data.py         — 训练数据结构、重复样本和哈希审计
eval.py                — 交互式推理/对话脚本
internlm2-1_8b-reward/ — InternLM2 Reward Model（用于 PPO/GRPO 的打分模型）
```

## 常用命令

### 训练

所有训练脚本通过命令行参数控制，无配置文件。

```bash
# 预训练
python trainer/train_pretrain.py --epochs 2 --batch_size 32 --learning_rate 5e-4 --use_moe 0

# 全参数 SFT（需先有 pretrain 权重）
python trainer/train_full_sft.py --epochs 2 --batch_size 16 --learning_rate 1e-6 --from_weight pretrain

# LoRA 微调（需先有 full_sft 权重）
python trainer/train_lora.py --epochs 50 --batch_size 32 --learning_rate 1e-4 --from_weight full_sft

# DPO
python trainer/train_dpo.py --epochs 1 --batch_size 4 --learning_rate 1e-6 --from_weight full_sft

# 推理模型蒸馏
python trainer/train_reason.py --epochs 2 --batch_size 8 --learning_rate 1e-6 --from_weight dpo

# PPO（需要 Reward Model 和 Ref Model）
python trainer/train_ppo.py --epochs 1 --batch_size 4 --learning_rate 1e-6 --reasoning 1

# GRPO（不需要 Critic，只需要 Reward Model）
python trainer/train_grpo.py --epochs 1 --batch_size 2 --learning_rate 1e-6 --reasoning 1
```

### 推理

```bash
python eval.py --weight dpo --hidden_size 512 --num_hidden_layers 8
```

### 评测

```bash
# 非 RL 管线 smoke test（CPU 友好）
python eval/smoke_test.py --all --skip-rl --device cpu

# 单个管线
python eval/smoke_test.py --stage pretrain

# 完整 smoke test（PPO/GRPO 需要本地 Reward Model，GPU 更合适）
python eval/smoke_test.py --all --device cuda:0

# 对已训练权重跑 benchmark
python eval/benchmark.py --weight dpo --stage all
python eval/benchmark.py --weight reason --stage format

# === 严格量化评估系统 ===

# 单个维度评估
python evals/eval_lm.py --checkpoint_path out/dpo_512.pth --data_path evals/data/lm_eval_sample.txt --device cuda
python evals/eval_qa.py --checkpoint_path out/full_sft_512.pth --data_path evals/data/qa_eval_sample.jsonl --do_sample
python evals/eval_generation.py --checkpoint_path out/full_sft_512.pth --data_path evals/data/generation_eval_sample.jsonl --do_sample
python evals/eval_speed.py --checkpoint_path out/dpo_512.pth --device cuda

# 一键运行全部 + 生成汇总报告
python evals/run_all.py --checkpoint_path out/dpo_512.pth --device cuda
bash scripts/run_eval.sh out/dpo_512.pth
```

### 关键参数速查

| 参数 | pretrain | SFT | LoRA | DPO | PPO/GRPO | Reason |
|------|----------|-----|------|-----|----------|--------|
| learning_rate | 5e-4 | 1e-6 | 1e-4 | 1e-6 | 1e-6 | 1e-6 |
| batch_size | 32 | 16 | 32 | 4 | PPO 4 / GRPO 2 | 8 |
| accumulation_steps | 8 | 8 | 1 | 1 | 1 | 1 |
| max_seq_len | 512 | 512 | 340 | 1024 | 66（另生成 512） | 1024 |
| epochs | 1-6 | 2 | 50 | 1 | 1 | 2 |

## 模型架构要点

- **配置类 `MiniMindConfig`** 继承 `PretrainedConfig`，`model_type = "minimind"`
- **默认 8 层 512 维**（约 26M 参数），可选 MoE（约 95.05M）或 16 层 768 维（Base，约 104M）
- **GQA**：`num_attention_heads=8`, `num_key_value_heads=2`
- **SwiGLU FFN**：`intermediate_size` 默认为 `hidden_size * 8/3` 并对齐到 64 的倍数
- **RoPE** 支持 YaRN 外推（`rope_scaling` config 非空时启用）
- **MoE** (`use_moe=True`)：`MoEGate` + `MoEFeedForward`，支持共享专家和负载均衡辅助损失
- **权重共享**：`embed_tokens.weight == lm_head.weight`
- **Flash Attention**：非缓存多 token 前向使用 `F.scaled_dot_product_attention`（因果 + padding mask）；缓存生成保留手写路径
- **ChatML 对话格式**：`<|im_start|>role\n...<|im_end|>\n`

## 训练架构

- **DDP 多卡训练**：`init_distributed_mode()` 读取 RANK/LOCAL_RANK 环境变量并使用 NCCL；多卡训练需 Linux + CUDA，原生 Windows 不支持 NCCL
- **混合精度**：bfloat16 优先（更稳定），float16 需 GradScaler
- **学习率调度**：`get_lr()` 实现 10% warmup + 90% cosine decay
- **断点续训**：`lm_checkpoint()` 保存/加载 model、optimizer、scaler、epoch、step 和 wandb_id；PPO/GRPO 另保存每 rank RNG 状态，DPO/PPO/GRPO 保存冻结参考模型 sidecar
- **安全恢复**：所有训练脚本默认 `--from_resume 0`；恢复时严格校验数据文件、batch、累积、序列长度、dtype、world size、compile 和模型架构，旧检查点需显式 `--allow_legacy_resume 1`
- **SkipBatchSampler**：续训时跳过已训练的 batch，确保数据遍历一致
- **动态 padding**：SFT/预训练等按批次最长样本补齐，并使用局部长度分桶降低 padding 浪费
- **loss mask**：SFT 只对 assistant 回复部分计算 loss（通过 `<|im_start|>assistant` 定位）；pretrain 对全部非 PAD token 计算
- **SwanLab**：国内 WandB 替代，API 兼容 `swanlab.init/log/get_run`

## 注意事项

- model 目录下的 tokenizer 使用 BPE 分词，vocab_size=6400，特殊 token：`<|im_start|>`=1, `<|im_end|>`=2, `<|endoftext|>`=0（PAD）
- 训练脚本通过 `trainer/path_utils.py` 将相对路径锚定到项目根；文档示例以项目根为 cwd，但使用绝对脚本路径时可从其他 cwd 启动
- `eval.py` 的 `--load_from` 默认 `"model"` 对应 `model/` 目录下的 tokenizer + 原生 torch 权重；其他值对应 transformers 格式
- Windows checkpoint 使用同目录临时文件加 `os.replace` 原子安装；目标被占用时保留旧文件并明确失败，不做非原子覆盖回退
- MoE 权重文件名加 `_moe` 后缀区分
- `eval.py` 会 shadow `eval/` 包目录，eval/ 内的脚本使用 `from eval_utils import ...`（非 `from eval.eval_utils import ...`）
