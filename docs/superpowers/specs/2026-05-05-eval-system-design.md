# MiniMind 评测系统设计

日期：2026-05-05

## 开发环境

- OS: Windows 11
- Python 环境: Conda 虚拟环境 `pytorch_env`，PyTorch/CUDA/transformers/datasets 已就绪
- 所有评测脚本在 `pytorch_env` 环境下运行

## 目标

构建闭环评测系统，覆盖两个层面：
1. **框架正确性**（smoke test）：快速验证训练管线无 bug
2. **模型质量**（benchmark）：评估已训练模型的各项能力

输出：终端 + JSON 报告 + Markdown 报告 + SwanLab 上报。

## 目录结构

```
eval/
├── smoke_test.py          # 框架正确性快速测试
├── benchmark.py           # 模型质量深度评估
├── eval_utils.py          # 共享工具（报告生成、SwanLab 集成、指标函数）
├── test_data/             # 小型 smoke 数据集
│   ├── pretrain_smoke.jsonl    # ~100 条纯文本
│   ├── sft_smoke.jsonl         # ~50 条 ChatML 对话
│   ├── dpo_smoke.jsonl         # ~30 条 chosen/rejected 对
│   ├── rlaif_smoke.jsonl       # ~20 条 prompt-answer 对
│   └── reason_smoke.jsonl      # ~20 条推理蒸馏数据
└── reports/               # 评测报告输出（gitignore）
```

## Smoke Test（框架正确性）

目标：每个训练管线跑 50 步即可判定管线完好。所有测试 CPU 可跑（无需 CUDA），PPO/GRPO 需要 Reward Model 的测试在无 GPU 时跳过。

### 通用检查项（每个测试共用）

- 模型初始化 → forward 不报错
- 梯度非零且不含 NaN
- 50 步后 loss 下降 > 10%（相对初始值）
- checkpoint save → load 后 forward 输出与保存前一致（allclose rtol=1e-3）

### 各阶段专项检查

| 阶段 | 专项断言 |
|------|---------|
| pretrain | 从零训练，loss 单调下降趋势 |
| SFT | 加载 pretrain 权重衔接正常；loss_mask 使 prompt 部分不贡献 loss |
| LoRA | 非 LoRA 参数 grad 为 None；保存的 lora.pth 只含 `lora.A/B` 权重 |
| DPO | chosen 平均 log-prob > rejected 平均 log-prob；DPO loss < ln(2)（随机基线） |
| Reason | tag penalty mask 命中 `<think>/<answer>` 标签，tag_hit_ratio > 0 |
| PPO | Actor/Critic/Ref/Reward 四个模型正常加载；rollout 完成不报错；GAE advantage 非全零 |
| GRPO | `num_generations` 个回答互不相同；组内 advantage std > 0 |

### CLI

```bash
# 全部管线 smoke test
python eval/smoke_test.py --all

# 单个管线
python eval/smoke_test.py --stage pretrain
python eval/smoke_test.py --stage sft

# 跳过需要 Reward Model 的测试
python eval/smoke_test.py --all --skip-rl
```

## Benchmark（模型质量）

需要已有一定训练量的模型权重文件才跑。通过 `--weight` 指定权重名，`--stage` 指定评测维度。

### 评测维度

1. **Perplexity**：在各阶段 test set 上计算 token-level PPL
2. **生成质量**：平均生成长度、重复 n-gram 比例（rep-4）、`<|im_end|>` 终止率、空响应率
3. **Reasoning 格式**：`<think>...</think><answer>...</answer>` 标签完整率、嵌套正确率
4. **Reward 分布**：用 Reward Model 给生成回答打分，输出均值/中位数/标准差
5. **分类准确率**：简单多选题（如有对应数据集）

### CLI

```bash
# 对 dpo 权重跑全部 benchmark
python eval/benchmark.py --weight dpo --stage all

# 只跑 reasoning 格式检查
python eval/benchmark.py --weight reason --stage format

# 指定模型尺寸
python eval/benchmark.py --weight full_sft --hidden_size 512 --num_hidden_layers 8
```

## 报告系统

### JSON 报告

`reports/{stage}_{timestamp}.json`：
```json
{
  "stage": "pretrain",
  "timestamp": "2026-05-05T10:30:00",
  "git_commit": "abc1234",
  "passed": true,
  "metrics": {
    "initial_loss": 8.52,
    "final_loss": 6.91,
    "loss_drop_pct": 18.9,
    "grad_norm_mean": 0.42,
    "checkpoint_roundtrip_allclose": true
  },
  "assertions": [
    {"name": "loss_drop_gt_10pct", "passed": true, "detail": "18.9% > 10%"},
    {"name": "grad_not_nan", "passed": true}
  ]
}
```

### Markdown 报告

`reports/{stage}_{timestamp}.md`：表格化展示所有指标和断言结果，含 git commit 信息。

### SwanLab 集成

- project: `MiniMind-Eval`
- 命名空间: `eval/{stage}/{metric_name}`
- 每次评测作为一个 step 上报，可画趋势曲线对比不同版本

## 接口设计

### `eval_utils.py` 提供的共享函数

```python
# 报告
def generate_report(stage, metrics, assertions, output_dir) -> dict  # 返回 metrics dict
def save_json_report(report, path)
def save_md_report(report, path)

# SwanLab
def init_swanlab(project="MiniMind-Eval")
def log_to_swanlab(stage, metrics, step)

# Smoke test 通用模板
def run_smoke_test(train_fn, stage_name, config, assertions) -> bool  # passed?

# 指标
def compute_perplexity(model, dataloader) -> float
def check_grad_flow(model) -> dict  # {grad_norm, has_nan}
def verify_checkpoint_roundtrip(model, save_path, sample_input) -> bool
```

### 依赖关系

- `smoke_test.py` 依赖 `trainer.*` 模块（复用训练脚本中的模型初始化、数据集类）
- `benchmark.py` 依赖 `eval.py` 中的 `init_model` 逻辑，以及 `internlm2-1_8b-reward/`
- 两者都依赖 `eval_utils.py`
- 不对现有训练脚本和模型代码做任何修改

## test_data 生成

每个 smoke 数据集从现有训练数据中抽取少量样本，或手写最小示例。优先从 `dataset/` 目录现有文件中 head 截取。

- `pretrain_smoke.jsonl`：100 条 `{"text": "..."}`
- `sft_smoke.jsonl`：50 条 `{"conversations": [...]}`
- `dpo_smoke.jsonl`：30 条 `{"chosen": [...], "rejected": [...]}`
- `rlaif_smoke.jsonl`：20 条 `{"conversations": [{"content": "..."}]}`
- `reason_smoke.jsonl`：20 条带 `<think>/<answer>` 标签的 SFT 格式数据

## 非目标

- 不做在线 eval（训练过程中周期性评测）。后续可加，本次只做离线评测。
- 不做多 GPU 分布式评测。全部单卡运行。
- 不修改现有训练脚本。评测系统是独立新增模块。
- 不做 HTML 报告。JSON + MD + SwanLab 已覆盖所需场景。
