# CC_Handmaking_LLM

从零实现的轻量级语言模型训练框架，纯 PyTorch，不依赖 transformers 模型类。

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

- 默认 8 层 512 维（~26M 参数），可选 MoE（~145M）或 16 层 768 维（~104M）
- GQA、SwiGLU FFN、RoPE+YaRN、Flash Attention
- ChatML 对话格式，vocab_size=6400

## 快速开始

```bash
# 预训练
python trainer/train_pretrain.py --epochs 2 --batch_size 32 --learning_rate 5e-4

# 全参数 SFT（需先有 pretrain 权重）
python trainer/train_full_sft.py --epochs 2 --batch_size 16 --learning_rate 1e-6 --from_weight pretrain

# 交互式推理
python eval.py --weight dpo
```

## 评测

```bash
# 框架正确性 smoke test（CPU 可跑）
python eval/smoke_test.py --all

# 模型质量 benchmark
python eval/benchmark.py --weight dpo --stage all

# 严格量化评估（一键全部）
python evals/run_all.py --checkpoint_path out/dpo_512.pth --device cuda

# 单个维度
python evals/eval_lm.py --checkpoint_path out/dpo_512.pth --data_path evals/data/lm_eval_sample.txt
python evals/eval_qa.py --checkpoint_path out/full_sft_512.pth --data_path evals/data/qa_eval_sample.jsonl --do_sample
python evals/eval_speed.py --checkpoint_path out/dpo_512.pth --device cuda
```

详见 `evals/README.md`。
