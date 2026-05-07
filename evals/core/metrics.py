"""量化指标计算函数

所有函数均返回 dict，key 为指标名，value 为数值。
函数签名统一为: fn(args...) -> dict
"""

import re
import math
import torch
import numpy as np


def normalize_text(text: str) -> str:
    """规范化文本用于 exact match 比较"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_qa_metrics(predictions: list, answers: list, keywords_list: list, categories: list) -> dict:
    """计算问答评估指标

    Args:
        predictions: 模型生成的答案列表
        answers: 参考答案列表
        keywords_list: 每个样本的关键词列表 (list of list)
        categories: 每个样本的类别列表

    Returns:
        dict 包含所有指标
    """
    n = len(predictions)
    exact_matches = 0
    kw_hits = 0
    kw_recalls = []
    non_empty = 0
    total_gen_len = 0

    cat_stats = {}  # {category: {"total": int, "exact": int, "kw_recall_sum": float}}

    for i in range(n):
        pred = predictions[i]
        ref = answers[i]
        kws = keywords_list[i] if i < len(keywords_list) else []
        cat = categories[i] if i < len(categories) else "unknown"

        pred_norm = normalize_text(pred)
        ref_norm = normalize_text(ref)

        if pred_norm == ref_norm:
            exact_matches += 1

        if len(pred.strip()) > 0:
            non_empty += 1

        total_gen_len += len(pred)

        hit_any = False
        hit_count = 0
        for kw in kws:
            if kw in pred:
                hit_count += 1
                hit_any = True
        if hit_any:
            kw_hits += 1
        recall = hit_count / max(len(kws), 1)
        kw_recalls.append(recall)

        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "exact": 0, "kw_recall_sum": 0.0, "kw_hits": 0}
        cat_stats[cat]["total"] += 1
        if pred_norm == ref_norm:
            cat_stats[cat]["exact"] += 1
        cat_stats[cat]["kw_recall_sum"] += recall
        if hit_any:
            cat_stats[cat]["kw_hits"] += 1

    metrics = {
        "total_cases": n,
        "exact_match": exact_matches,
        "exact_match_rate": exact_matches / max(n, 1),
        "keyword_hit_rate": kw_hits / max(n, 1),
        "average_keyword_recall": np.mean(kw_recalls) if kw_recalls else 0.0,
        "answer_non_empty_rate": non_empty / max(n, 1),
        "average_generation_length": total_gen_len / max(n, 1),
    }

    for cat, stats in cat_stats.items():
        t = stats["total"]
        metrics[f"category_{cat}_count"] = t
        metrics[f"category_{cat}_exact_match_rate"] = stats["exact"] / max(t, 1)
        metrics[f"category_{cat}_keyword_recall"] = stats["kw_recall_sum"] / max(t, 1)
        metrics[f"category_{cat}_keyword_hit_rate"] = stats["kw_hits"] / max(t, 1)

    return metrics


def compute_generation_constraint_metrics(results: list) -> dict:
    """计算生成约束评估指标

    Args:
        results: list of dict, 每个 dict 包含:
            - format_correct: bool
            - json_parsed: bool
            - required_keys_ok: bool
            - length_ok: bool
            - must_include_ok: bool
            - forbid_include_ok: bool

    Returns:
        dict 包含所有指标
    """
    n = len(results)
    format_ok = sum(1 for r in results if r.get("format_correct", False))
    json_ok = sum(1 for r in results if r.get("json_parsed", False))
    keys_ok = sum(1 for r in results if r.get("required_keys_ok", False))
    length_ok = sum(1 for r in results if r.get("length_ok", False))
    must_ok = sum(1 for r in results if r.get("must_include_ok", False))
    forbid_ok = sum(1 for r in results if r.get("forbid_include_ok", False))
    overall = sum(
        1 for r in results
        if r.get("format_correct", False)
        and r.get("json_parsed", False)
        and r.get("required_keys_ok", False)
        and r.get("length_ok", False)
        and r.get("must_include_ok", False)
        and r.get("forbid_include_ok", False)
    )

    return {
        "total_cases": n,
        "format_success_rate": format_ok / max(n, 1),
        "json_parse_success_rate": json_ok / max(n, 1),
        "required_key_success_rate": keys_ok / max(n, 1),
        "length_constraint_success_rate": length_ok / max(n, 1),
        "must_include_success_rate": must_ok / max(n, 1),
        "forbid_include_success_rate": forbid_ok / max(n, 1),
        "overall_constraint_success_rate": overall / max(n, 1),
    }


def compute_speed_metrics(
    latencies_ms: list,
    total_generated_tokens: int,
    total_prompt_tokens: int,
    peak_memory_mb: float = None,
    device: str = "cpu",
    dtype: str = "fp32",
) -> dict:
    """计算推理速度指标

    Args:
        latencies_ms: 每次推理的延迟列表（毫秒）
        total_generated_tokens: 总生成 token 数
        total_prompt_tokens: 总 prompt token 数
        peak_memory_mb: GPU 峰值显存（MB），CPU 环境可为 None
        device: 设备
        dtype: 数据类型

    Returns:
        dict 包含所有指标
    """
    arr = np.array(latencies_ms)
    total_time_s = arr.sum() / 1000.0

    metrics = {
        "average_latency_ms": float(np.mean(arr)),
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "tokens_per_second": total_generated_tokens / max(total_time_s, 0.001),
        "generated_tokens": total_generated_tokens,
        "prompt_tokens": total_prompt_tokens,
        "peak_gpu_memory_mb": peak_memory_mb,
        "device": device,
        "dtype": dtype,
        "num_runs": len(latencies_ms),
        "total_time_seconds": total_time_s,
    }
    return metrics


def compute_lm_metrics(
    total_loss: float,
    total_tokens: int,
    num_samples: int,
    total_sequence_length: int,
    max_sequence_length: int,
    batch_size: int,
    device: str,
    eval_time_seconds: float,
) -> dict:
    """计算语言模型评估指标

    Args:
        total_loss: 累积 cross entropy loss * token_count
        total_tokens: 参与 loss 计算的 token 总数
        num_samples: 评估样本数
        total_sequence_length: 所有样本的总 token 数（含 padding）
        max_sequence_length: 最大序列长度
        batch_size: 批大小
        device: 设备
        eval_time_seconds: 评估耗时（秒）

    Returns:
        dict 包含所有指标
    """
    val_loss = total_loss / max(total_tokens, 1)
    try:
        ppl = math.exp(min(val_loss, 20.0))
    except OverflowError:
        ppl = float("inf")

    return {
        "validation_loss": val_loss,
        "perplexity": ppl,
        "evaluated_tokens": total_tokens,
        "evaluated_samples": num_samples,
        "average_sequence_length": total_sequence_length / max(num_samples, 1),
        "max_sequence_length": max_sequence_length,
        "batch_size": batch_size,
        "device": device,
        "eval_time_seconds": eval_time_seconds,
    }
