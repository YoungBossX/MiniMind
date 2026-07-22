# MiniMind 严格量化评估系统

对 MiniMind 模型进行多维度量化评估，支持横向对比、自动报告生成。

## 目录结构

```
evals/
  core/
    __init__.py       — 包入口
    load_model.py     — 统一模型/Tokenizer/Checkpoint 加载
    metrics.py        — 所有量化指标计算函数
    io_utils.py       — JSONL/TXT/JSON 读写工具
    report.py         — 评估结果生成 Markdown 报告
  configs/
    eval_config.yaml  — 默认配置文件
  data/
    lm_eval_sample.txt          — LM 评估样例数据
    qa_eval_sample.jsonl         — QA 评估样例数据
    generation_eval_sample.jsonl — 约束生成样例数据
  eval_lm.py          — 语言模型评估（PPL / Validation Loss）
  eval_qa.py          — 领域问答评估（Exact Match / Keyword Recall）
  eval_generation.py  — 生成约束评估（JSON / 长度 / 关键词）
  eval_speed.py       — 推理性能评估（tokens/s / 延迟 / 显存）
  run_all.py          — 一键运行全部评估
  compare_runs.py     — 对比两次评估输出的指标变化
```

## 快速开始

### 单个评估模块

```bash
# 语言模型评估
python evals/eval_lm.py \
  --checkpoint_path out/dpo_512.pth \
  --data_path evals/data/lm_eval_sample.txt \
  --output_path outputs/evals/lm_eval.json \
  --device cuda --batch_size 4 --max_length 512

# 问答评估
python evals/eval_qa.py \
  --checkpoint_path out/full_sft_512.pth \
  --data_path evals/data/qa_eval_sample.jsonl \
  --output_path outputs/evals/qa_eval.json

# 约束生成评估
python evals/eval_generation.py \
  --checkpoint_path out/full_sft_512.pth \
  --data_path evals/data/generation_eval_sample.jsonl \
  --output_path outputs/evals/generation_eval.json

# 推理速度评估
python evals/eval_speed.py \
  --checkpoint_path out/dpo_512.pth \
  --output_path outputs/evals/speed_eval.json
```

### 一键运行全部

```bash
# 使用命令行参数
python evals/run_all.py --checkpoint_path out/dpo_512.pth

# 使用配置文件
python evals/run_all.py --checkpoint_path out/dpo_512.pth --config_path evals/configs/eval_config.yaml

# 跳过某些评估
python evals/run_all.py --checkpoint_path out/dpo_512.pth --skip_speed
```

`--checkpoint_path` 默认必须提供；只有明确传入 `--allow_random_init` 才会使用
随机初始化模型。后者只适合检查评估编排，不适合解读 QA 或生成质量。

### Shell 脚本

```bash
bash scripts/run_eval.sh
bash scripts/run_eval.sh out/dpo_512.pth
```

该脚本只调用一次 `run_all.py`，失败会保留非零退出码。

### 对比两次运行

```bash
python evals/compare_runs.py \
  --baseline_dir outputs/evals-baseline \
  --current_dir outputs/evals \
  --output_path outputs/evals/compare_runs.json
```

## 评估维度与指标

| 模块 | 指标 | 说明 |
|------|------|------|
| **LM Eval** | validation_loss, perplexity, evaluated_tokens | token-level 语言建模能力 |
| **QA Eval** | exact_match_rate, keyword_hit_rate, average_keyword_recall | 领域知识问答准确率 |
| **Generation** | format_success_rate, json_parse_success_rate, length_constraint_success_rate | 结构化输出约束遵守能力 |
| **Speed** | average_latency_ms, p50/p95_latency_ms, tokens_per_second, peak_gpu_memory_mb | 推理性能与资源占用 |

## 输出文件

默认结果输出到 `outputs/evals/`；可用 `run_all.py --output_dir` 或配置文件中的
`eval.output_dir` 指定其他目录：

- `lm_eval.json` — 语言模型评估指标
- `qa_eval.json` — 问答评估指标
- `qa_predictions.jsonl` — 每条样本的预测详情
- `generation_eval.json` — 生成约束评估指标
- `generation_predictions.jsonl` — 每条约束样本的检查结果
- `speed_eval.json` — 推理性能指标
- `eval_report.md` — 汇总 Markdown 报告（run_all.py 生成）
- `compare_runs.json` — 两次评估的指标差值与 improved/regressed 状态

## 自定义评估数据

创建对应格式的 JSONL 或 TXT 文件后，通过 `--data_path` 参数指定即可：

- LM 评估：每行一条文本的 TXT 文件
- QA 评估：`{"id": "...", "question": "...", "answer": "...", "keywords": [...], "category": "..."}` 的 JSONL
- 约束生成：`{"id": "...", "prompt": "...", "constraints": {...}}` 的 JSONL
