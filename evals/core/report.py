"""根据评估 JSON 结果生成 Markdown 报告"""

import os
import json
from datetime import datetime, timezone


def generate_markdown_report(
    checkpoint_path: str,
    all_metrics: dict,
    output_dir: str,
    device: str = "cpu",
    dtype: str = "fp32",
    seed: int = 42,
    failed_samples: dict = None,
    stage_status: dict = None,
) -> str:
    """生成 eval_report.md

    Args:
        checkpoint_path: 模型权重路径
        all_metrics: 所有评估模块的指标 dict（嵌套结构）
        output_dir: 输出目录
        device: 设备
        dtype: 数据类型
        seed: 随机种子
        failed_samples: 按模块分组的失败样例 {"lm": [...], "qa": [...], "gen": [...]}

    Returns:
        报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    lines = [
        "# MiniMind Quantitative Evaluation Report",
        "",
        "## 1. Evaluation Setup",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Checkpoint | `{checkpoint_path}` |",
        f"| Device | `{device}` |",
        f"| Dtype | `{dtype}` |",
        f"| Seed | {seed} |",
        f"| Evaluation Time | {timestamp} |",
        "",
    ]

    if stage_status is not None:
        lines += [
            "## Evaluation Status",
            "",
            "| Stage | Status |",
            "|-------|--------|",
        ]
        for stage, status in stage_status.items():
            lines.append(f"| {stage} | {status} |")
        lines.append("")

    def _add_metric_section(title, metrics_dict, indent_level=2):
        nonlocal lines
        prefix = "#" * indent_level
        lines += [
            f"{prefix} {title}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if metrics_dict:
            for k, v in metrics_dict.items():
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.6f} |")
                else:
                    lines.append(f"| {k} | {v} |")
        else:
            lines.append("| (no data) | — |")
        lines.append("")

    _add_metric_section("2. Language Modeling Metrics", all_metrics.get("lm", {}))
    _add_metric_section("3. QA Metrics", all_metrics.get("qa", {}))
    _add_metric_section("4. Generation Constraint Metrics", all_metrics.get("generation", {}))
    _add_metric_section("5. Inference Speed Metrics", all_metrics.get("speed", {}))

    # Error Analysis
    lines += [
        "## 6. Error Analysis",
        "",
    ]

    if failed_samples:
        printed = 0
        for module, samples in failed_samples.items():
            if samples:
                lines.append(f"### {module.upper()} Failures")
                lines.append("")
                for s in samples[:10]:
                    lines.append(f"- **id={s.get('id', '?')}**")
                    if "question" in s:
                        lines.append(f"  - Question: {s['question'][:100]}")
                    if "prompt" in s:
                        lines.append(f"  - Prompt: {s['prompt'][:100]}")
                    if "prediction" in s:
                        lines.append(f"  - Prediction: {str(s['prediction'])[:200]}")
                    if "reason" in s:
                        lines.append(f"  - Reason: {s['reason']}")
                    lines.append("")
                printed += 1
        if printed == 0:
            lines.append("No error analysis data available.")
            lines.append("")
    else:
        lines.append("No error analysis data available.")
        lines.append("")

    # Optimization Suggestions
    lines += [
        "## 7. Optimization Suggestions",
        "",
    ]
    suggestions = _generate_suggestions(all_metrics)
    for s in suggestions:
        lines.append(f"- {s}")

    report_path = os.path.join(output_dir, "eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Report] Markdown report saved: {report_path}")
    return report_path


def _generate_suggestions(all_metrics: dict) -> list:
    """根据指标自动生成优化建议"""
    suggestions = []

    lm = all_metrics.get("lm", {})
    qa = all_metrics.get("qa", {})
    gen = all_metrics.get("generation", {})
    speed = all_metrics.get("speed", {})

    ppl = lm.get("perplexity", None)
    if ppl is not None and ppl > 50:
        suggestions.append(
            f"PPL 较高 ({ppl:.1f})：检查预训练语料质量、tokenizer 一致性和上下文长度设置。"
        )
    elif ppl is not None and ppl > 20:
        suggestions.append(
            f"PPL 偏高 ({ppl:.1f})：可考虑增加预训练数据量或提升数据多样性。"
        )

    kw_recall = qa.get("average_keyword_recall", None)
    if kw_recall is not None and kw_recall < 0.5:
        suggestions.append(
            f"Keyword Recall 较低 ({kw_recall:.2f})：补充领域指令数据、优化 SFT 数据格式和覆盖范围。"
        )

    json_rate = gen.get("json_parse_success_rate", None)
    if json_rate is not None and json_rate < 0.8:
        suggestions.append(
            f"JSON 解析成功率低 ({json_rate:.2f})：增加结构化输出训练数据或考虑约束解码策略。"
        )

    format_rate = gen.get("format_success_rate", None)
    if format_rate is not None and format_rate < 0.7:
        suggestions.append(
            f"格式约束成功率低 ({format_rate:.2f})：增加格式约束的训练数据，强化模型遵循指令的能力。"
        )

    tps = speed.get("tokens_per_second", None)
    if tps is not None and tps < 10:
        suggestions.append(
            f"tokens/s 较低 ({tps:.1f})：检查是否启用 Flash Attention、调整 batch size 或 dtype。"
        )

    if not suggestions:
        suggestions.append("当前指标表现正常，可继续优化训练数据和模型架构以进一步提升性能。")

    return suggestions
