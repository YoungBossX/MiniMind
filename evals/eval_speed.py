"""推理性能评估 — tokens/s / 延迟 / 显存占用

评估模型推理速度和资源消耗。

用法:
    python evals/eval_speed.py --checkpoint_path out/pretrain_512.pth
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.core.load_model import load_model_and_tokenizer
from evals.core.metrics import compute_speed_metrics
from evals.core.io_utils import write_json


def evaluate_speed(model, tokenizer, device="cpu", max_new_tokens=128,
                   warmup_runs=3, repeat_runs=10, batch_size=1):
    """评估推理速度"""
    test_prompts = [
        "你好，请介绍一下深度学习。",
        "请用Python写一个快速排序算法。",
        "Explain what is machine learning.",
    ]

    latencies_ms = []
    total_gen_tokens = 0
    total_prompt_tokens = 0

    model.eval()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for run_idx in range(warmup_runs + repeat_runs):
        prompt = test_prompts[run_idx % len(test_prompts)]
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=128, return_token_type_ids=False).to(device)

        p_tokens = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        g_tokens = gen.shape[1] - p_tokens

        if run_idx >= warmup_runs:
            latencies_ms.append(elapsed_ms)
            total_gen_tokens += g_tokens
            total_prompt_tokens += p_tokens

    peak_memory_mb = None
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return compute_speed_metrics(
        latencies_ms=latencies_ms,
        total_generated_tokens=total_gen_tokens,
        total_prompt_tokens=total_prompt_tokens,
        peak_memory_mb=peak_memory_mb,
        device=device,
        dtype=str(model.dtype) if hasattr(model, "dtype") else "fp32",
    )


def main():
    parser = argparse.ArgumentParser(description="MiniMind Inference Speed Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default="", help="模型权重 .pth 路径")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer 目录路径")
    parser.add_argument("--output_path", type=str, default="outputs/evals/speed_eval.json", help="输出 JSON 路径")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--warmup_runs", type=int, default=3)
    parser.add_argument("--repeat_runs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.checkpoint_path:
        print("[WARNING] checkpoint_path 为空，将使用随机初始化模型。")

    print(f"Loading model from: {args.checkpoint_path or '(random init)'}")
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        device=args.device,
        dtype=args.dtype,
    )

    print(f"Running speed benchmark: warmup={args.warmup_runs}, repeat={args.repeat_runs}")
    metrics = evaluate_speed(
        model, tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        repeat_runs=args.repeat_runs,
        batch_size=args.batch_size,
    )

    result = {
        "task": "speed_eval",
        "checkpoint": args.checkpoint_path,
        **metrics,
    }

    write_json(args.output_path, result)

    print(f"Average Latency: {metrics['average_latency_ms']:.1f}ms")
    print(f"P50 Latency:     {metrics['p50_latency_ms']:.1f}ms")
    print(f"P95 Latency:     {metrics['p95_latency_ms']:.1f}ms")
    print(f"Tokens/s:        {metrics['tokens_per_second']:.1f}")
    print(f"Peak GPU Mem:    {metrics['peak_gpu_memory_mb']}")
    print(f"Total Time:      {metrics['total_time_seconds']:.1f}s ({metrics['num_runs']} runs)")


if __name__ == "__main__":
    main()
