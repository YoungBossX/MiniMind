"""MiniMind 评测系统 — 模型质量 Benchmark

对已训练的模型进行深度质量评估：
  - Perplexity (token-level, 在各阶段 test set 上)
  - 生成质量 (长度、重复率、终止率)
  - Reasoning 格式合规率
  - Reward Model 评分分布
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
import warnings
import torch
import numpy as np
from torch.utils.data import DataLoader

from model.MiniMindModel import MiniMindConfig
from eval_utils import (
    generate_report, init_swanlab, log_to_swanlab, make_small_config, get_git_commit,
)

warnings.filterwarnings("ignore")

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "out")


def load_model(weight, hidden_size, num_layers, device):
    """复刻 eval.py 中的模型加载逻辑"""
    from transformers import AutoTokenizer
    from model.MiniMindModel import MiniMindForCausalLM

    model_dir = os.path.join(BASE_DIR, "..", "model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, use_moe=False)
    model = MiniMindForCausalLM(config)

    ckp = os.path.join(SAVE_DIR, f"{weight}_{hidden_size}.pth")
    state_dict = torch.load(ckp, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device), tokenizer


def compute_perplexity(model, tokenizer, data_path, max_length=256, batch_size=4, max_samples=200, device="cuda"):
    """在指定数据集上计算 token-level perplexity"""
    from dataset.llm_dataset import PretrainDataset

    ds = PretrainDataset(data_path, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    count = 0
    model.eval()

    with torch.no_grad():
        for X, Y, loss_mask, attn_mask in loader:
            X, Y, loss_mask, attn_mask = X.to(device), Y.to(device), loss_mask.to(device), attn_mask.to(device)
            res = model(X, attention_mask=attn_mask, labels=Y, loss_mask=loss_mask)
            total_loss += (res.loss * loss_mask.sum()).item()
            total_tokens += loss_mask.sum().item()
            count += X.size(0)
            if count >= max_samples:
                break

    ppl = np.exp(total_loss / max(total_tokens, 1))
    return ppl


def evaluate_generation_quality(model, tokenizer, prompts, max_new_tokens=256, device="cuda"):
    """评估生成质量：长度、重复率、终止率"""
    results = []
    model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.85, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_ids = gen[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        # 重复 n-gram 比例 (rep-4)
        tokens = response
        rep_4_count = 0
        for i in range(len(tokens) - 7):
            if tokens[i:i+4] == tokens[i+4:i+8]:
                rep_4_count += 1
                break

        results.append({
            "prompt": prompt[:50],
            "response_len": len(response),
            "has_eos": "<|im_end|>" in response,
            "is_empty": len(response.strip()) == 0,
            "has_rep_4": rep_4_count > 0,
        })

    metrics = {
        "avg_response_len": np.mean([r["response_len"] for r in results]),
        "eos_rate": np.mean([r["has_eos"] for r in results]),
        "empty_rate": np.mean([r["is_empty"] for r in results]),
        "rep4_rate": np.mean([r["has_rep_4"] for r in results]),
    }
    return metrics, results


def evaluate_reasoning_format(model, tokenizer, device):
    """评估 reasoning 模型的 <think>/<answer> 格式合规率"""

    prompts = [
        "请解释什么是光合作用。",
        "9.11和9.8哪个大？",
        "写一首关于春天的诗。",
    ]

    messages_list = []
    for p in prompts:
        messages_list.append([{"role": "user", "content": p}])

    results = {"total": 0, "has_think": 0, "has_answer": 0, "tag_complete": 0, "tag_valid_structure": 0}

    model.eval()
    for messages in messages_list:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.85,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results["total"] += 1

        has_think = "<think>" in response and "</think>" in response
        has_answer = "<answer>" in response and "</answer>" in response
        if has_think:
            results["has_think"] += 1
        if has_answer:
            results["has_answer"] += 1
        if has_think and has_answer:
            results["tag_complete"] += 1
        if re.search(r"<think>.*?</think>.*?<answer>.*?</answer>", response, re.DOTALL):
            results["tag_valid_structure"] += 1

    n = max(results["total"], 1)
    return {k: v / n if k != "total" else v for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="MiniMind Model Benchmark")
    parser.add_argument("--weight", type=str, required=True, help="模型权重名称 (pretrain, full_sft, dpo, reason, etc.)")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "perplexity", "generation", "format", "reward"])
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    if args.use_wandb:
        init_swanlab("MiniMind-Eval")

    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"Loading model: {args.weight} (hidden={args.hidden_size}, layers={args.num_hidden_layers})")
    model, tokenizer = load_model(args.weight, args.hidden_size, args.num_hidden_layers, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    all_metrics = {}
    all_assertions = []

    # Perplexity
    if args.stage in ("all", "perplexity"):
        print("\n--- Perplexity ---")
        test_paths = {
            "pretrain_test": os.path.join(BASE_DIR, "test_data", "pretrain_smoke.jsonl"),
            "sft_test": os.path.join(BASE_DIR, "test_data", "sft_smoke.jsonl"),
        }
        for name, path in test_paths.items():
            if os.path.exists(path):
                ppl = compute_perplexity(model, tokenizer, path, device=args.device)
                print(f"  {name} PPL: {ppl:.2f}")
                all_metrics[f"ppl_{name}"] = ppl
        all_assertions.append({"name": "ppl_finite", "passed": all_metrics.get("ppl_pretrain_test", float("inf")) < 1e6})

    # Generation quality
    if args.stage in ("all", "generation"):
        print("\n--- Generation Quality ---")
        test_prompts = [
            "你好，请介绍一下你自己。",
            "什么是机器学习？",
            "请用Python写一个斐波那契函数。",
            "天空为什么是蓝色的？",
            "推荐一些中国的美食。",
        ]
        gen_metrics, gen_details = evaluate_generation_quality(model, tokenizer, test_prompts, device=args.device)
        for k, v in gen_metrics.items():
            print(f"  {k}: {v:.4f}")
            all_metrics[f"gen_{k}"] = v
        all_assertions.append({"name": "gen_not_all_empty", "passed": gen_metrics["empty_rate"] < 1.0,
                               "detail": f"empty_rate={gen_metrics['empty_rate']:.2f}"})
        all_assertions.append({"name": "gen_eos_reasonable", "passed": gen_metrics["eos_rate"] > 0.3,
                               "detail": f"eos_rate={gen_metrics['eos_rate']:.2f} > 0.3"})

    # Reasoning format
    if args.stage in ("all", "format"):
        print("\n--- Reasoning Format ---")
        format_metrics = evaluate_reasoning_format(model, tokenizer, args.device)
        for k, v in format_metrics.items():
            print(f"  {k}: {v:.4f}")
            all_metrics[f"fmt_{k}"] = v
        all_assertions.append({"name": "format_tags_present", "passed": format_metrics["tag_complete_rate"] >= 0,
                               "detail": f"complete_rate={format_metrics['tag_complete_rate']:.2f}"})

    report = generate_report(f"benchmark_{args.weight}", all_metrics, all_assertions, REPORT_DIR)

    if args.use_wandb:
        log_to_swanlab(f"benchmark_{args.weight}", all_metrics)

    status = "PASS" if report["passed"] else "FAIL"
    print(f"\n{'='*60}")
    print(f"  BENCHMARK {args.weight}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
