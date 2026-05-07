"""生成质量规则评估 — 格式约束 / JSON 解析 / 长度 / must_include / forbid_include

对 JSONL 格式的约束生成数据进行评估。

用法:
    python evals/eval_generation.py --checkpoint_path out/full_sft_512.pth --data_path evals/data/generation_eval_sample.jsonl
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.core.load_model import load_model_and_tokenizer
from evals.core.metrics import compute_generation_constraint_metrics
from evals.core.io_utils import read_jsonl, write_json, write_jsonl


def check_constraints(prediction, constraints):
    """检查预测结果是否满足约束条件

    Returns:
        dict: {format_correct, json_parsed, required_keys_ok, length_ok,
               must_include_ok, forbid_include_ok, details}
    """
    result = {
        "format_correct": True,
        "json_parsed": True,
        "required_keys_ok": True,
        "length_ok": True,
        "must_include_ok": True,
        "forbid_include_ok": True,
        "details": [],
    }
    parsed = None

    min_len = constraints.get("min_length", 0)
    max_len = constraints.get("max_length", 99999)
    must_include = constraints.get("must_include", [])
    forbid_include = constraints.get("forbid_include", [])
    json_required = constraints.get("json_required", False)
    required_keys = constraints.get("required_keys", [])

    # 长度检查
    text_len = len(prediction)
    if text_len < min_len:
        result["length_ok"] = False
        result["format_correct"] = False
        result["details"].append(f"length={text_len} < min_length={min_len}")
    if text_len > max_len:
        result["length_ok"] = False
        result["format_correct"] = False
        result["details"].append(f"length={text_len} > max_length={max_len}")

    # must_include 检查
    for phrase in must_include:
        if phrase not in prediction:
            result["must_include_ok"] = False
            result["format_correct"] = False
            result["details"].append(f"missing required phrase: '{phrase}'")

    # forbid_include 检查
    for phrase in forbid_include:
        if phrase in prediction:
            result["forbid_include_ok"] = False
            result["format_correct"] = False
            result["details"].append(f"found forbidden phrase: '{phrase}'")

    # JSON 检查
    if json_required or required_keys:
        try:
            # 尝试从生成文本中提取 JSON
            json_start = prediction.find("{")
            json_end = prediction.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found")
            parsed = json.loads(prediction[json_start:json_end])
        except Exception:
            result["json_parsed"] = False
            result["format_correct"] = False
            result["details"].append("failed to parse JSON from output")
            parsed = {}

        if result["json_parsed"] and required_keys:
            for key in required_keys:
                if key not in parsed:
                    result["required_keys_ok"] = False
                    result["format_correct"] = False
                    result["details"].append(f"missing required key: '{key}'")
    else:
        result["json_parsed"] = True
        result["required_keys_ok"] = True

    return result


def evaluate_generation(model, tokenizer, data_path, device="cpu", max_new_tokens=256,
                        temperature=0.7, top_p=0.9, do_sample=False):
    """在约束生成数据集上评估"""
    samples = read_jsonl(data_path)
    if not samples:
        raise ValueError(f"No valid generation samples found in: {data_path}")

    constraint_results = []
    details = []

    model.eval()
    for s in samples:
        prompt = s["prompt"]
        constraints = s.get("constraints", {})

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=128, return_token_type_ids=False).to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_ids = gen[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(response_ids, skip_special_tokens=True)

        check = check_constraints(prediction, constraints)
        constraint_results.append(check)
        details.append({
            "id": s["id"],
            "prompt": prompt,
            "prediction": prediction,
            "constraints": constraints,
            **check,
        })

    metrics = compute_generation_constraint_metrics(constraint_results)
    return metrics, details


def main():
    parser = argparse.ArgumentParser(description="MiniMind Generation Constraint Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default="", help="模型权重 .pth 路径")
    parser.add_argument("--data_path", type=str, required=True, help="约束生成 JSONL 数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer 目录路径")
    parser.add_argument("--output_path", type=str, default="outputs/evals/generation_eval.json", help="输出 JSON 路径")
    parser.add_argument("--predictions_path", type=str, default="outputs/evals/generation_predictions.jsonl", help="预测结果 JSONL")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
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

    print(f"Evaluating on: {args.data_path}")
    metrics, details = evaluate_generation(
        model, tokenizer, args.data_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    result = {
        "task": "generation_eval",
        "checkpoint": args.checkpoint_path,
        "data_path": args.data_path,
        **metrics,
    }

    write_json(args.output_path, result)
    write_jsonl(args.predictions_path, details)

    print(f"Format Success Rate:      {metrics['format_success_rate']:.4f}")
    print(f"JSON Parse Success Rate:   {metrics['json_parse_success_rate']:.4f}")
    print(f"Required Key Success Rate: {metrics['required_key_success_rate']:.4f}")
    print(f"Length Constraint Rate:    {metrics['length_constraint_success_rate']:.4f}")
    print(f"Must Include Rate:         {metrics['must_include_success_rate']:.4f}")
    print(f"Forbid Include Rate:       {metrics['forbid_include_success_rate']:.4f}")
    print(f"Overall Constraint Rate:   {metrics['overall_constraint_success_rate']:.4f}")


if __name__ == "__main__":
    main()
