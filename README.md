# CC_Handmaking_LLM

从零实现的轻量级语言模型训练框架。模型主体使用 PyTorch 从零实现，不使用
transformers 内置模型架构；项目仍依赖 transformers 的 `PretrainedConfig`、
`PreTrainedModel`、`GenerationMixin`、标准输出类型和 tokenizer 接口。

## 训练管线

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 预训练 | `trainer/train_pretrain.py` | Next-Token Prediction |
| SFT | `trainer/train_full_sft.py` | 全参数监督微调 |
| LoRA | `trainer/train_lora.py` | 低秩适配微调 |
| DPO | `trainer/train_dpo.py` | 直接偏好优化 |
| PPO | `trainer/train_ppo.py` | Actor-Critic 强化学习 |
| GRPO | `trainer/train_grpo.py` | 无 Critic 的简化 RL |
| Reason | `trainer/train_reason.py` | 推理模型蒸馏 |

## 模型架构

- 默认 8 层 512 维（~26M 参数），可选 MoE（~95.05M）或 16 层 768 维（~104M）
- GQA、SwiGLU FFN、RoPE+YaRN；非缓存多 token 前向使用 PyTorch SDPA
- 动态 padding 与长度分桶减少短样本批次的无效计算
- ChatML 对话格式，vocab_size=6400

## 快速开始

```bash
# 预训练
python trainer/train_pretrain.py --epochs 2 --batch_size 32 --learning_rate 5e-4

# 全参数 SFT（需先有 pretrain 权重）
python trainer/train_full_sft.py --epochs 2 --batch_size 16 --learning_rate 1e-6 --max_seq_len 512 --from_weight pretrain

# 交互式推理
python eval.py --weight dpo
```

训练脚本默认 `--from_resume 0`，不会误接旧实验。新检查点使用 metadata
schema v2；显式传入 `--from_resume 1` 后，会校验数据 SHA-256、Tokenizer/模型
运行时内容、模型架构、训练目标参数、batch、dtype、world size 和 compile 设置，
PPO/GRPO 还会恢复每个 rank 的 RNG 状态。DPO/PPO/GRPO 的冻结参考模型保存在
同目录的内容寻址 `_ref_<sha256>.pth` sidecar 中，移动检查点时必须一并保留。

Tokenizer 或 Reward Model 若只填写远程 Hugging Face Hub ID，后续无法进行内容级
精确恢复；应改用本地不可变快照。只有明确接受非精确续训或兼容旧检查点时，才加
`--allow_legacy_resume 1`。

## 评测

```bash
# 框架正确性 smoke test（CPU 可跑）
python eval/smoke_test.py --all

# 模型质量 benchmark
python eval/benchmark.py --weight dpo --stage all

# 查看一键评估的完整参数
python evals/run_all.py --help

# 严格量化评估（一键全部，默认必须显式提供检查点）
python evals/run_all.py --checkpoint_path out/dpo_512.pth --device cuda

# 仅用于 smoke test 的随机初始化评估
python evals/run_all.py --allow_random_init --device cpu

# 单个维度
python evals/eval_lm.py --checkpoint_path out/dpo_512.pth --data_path evals/data/lm_eval_sample.txt
python evals/eval_qa.py --checkpoint_path out/full_sft_512.pth --data_path evals/data/qa_eval_sample.jsonl --do_sample
python evals/eval_speed.py --checkpoint_path out/dpo_512.pth --device cuda
```

`evals/run_all.py` 默认要求 `--checkpoint_path`。`--allow_random_init` 只用于
smoke test，随机初始化模型的 QA、生成等质量指标不具备参考价值。

详见 `evals/README.md`。
