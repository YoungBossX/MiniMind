"""领域问答评估 — Exact Match / Keyword Recall / 按类别统计

对 JSONL 格式的 QA 数据进行评估。

用法:
    python evals/eval_qa.py --checkpoint_path out/full_sft_512.pth --data_path evals/data/qa_eval_sample.jsonl
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.core.load_model import load_model_and_tokenizer
from evals.core.metrics import compute_qa_metrics
from evals.core.io_utils import read_jsonl, write_json, write_jsonl


def evaluate_qa(model, tokenizer, data_path, device="cpu", max_new_tokens=256,
                temperature=0.7, top_p=0.9, do_sample=False):
    """在 QA 数据集上评估"""
    samples = read_jsonl(data_path)
    if not samples:
        raise ValueError(f"No valid QA samples found in: {data_path}")

    predictions = []
    answers = []
    keywords_list = []
    categories = []

    model.eval()
    for s in samples:
        question = s["question"]
        answer = s.get("answer", "")
        keywords = s.get("keywords", [])
        category = s.get("category", "unknown")

        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128, return_token_type_ids=False).to(device)

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

        predictions.append(prediction)
        answers.append(answer)
        keywords_list.append(keywords)
        categories.append(category)

    metrics = compute_qa_metrics(predictions, answers, keywords_list, categories)

    details = []
    for i, s in enumerate(samples):
        pred = predictions[i]
        ref = answers[i]
        exact = pred.strip() == ref.strip() or " ".join(pred.split()) == " ".join(ref.split())
        kw_hits = [kw for kw in (keywords_list[i] if i < len(keywords_list) else []) if kw in pred]
        details.append({
            "id": s["id"],
            "question": s["question"],
            "prediction": pred,
            "reference": ref,
            "exact_match": exact,
            "keywords_expected": keywords_list[i] if i < len(keywords_list) else [],
            "keywords_hit": kw_hits,
            "category": categories[i] if i < len(categories) else "unknown",
        })

    return metrics, details


def main():
    parser = argparse.ArgumentParser(description="MiniMind QA Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default="", help="模型权重 .pth 路径")
    parser.add_argument("--data_path", type=str, required=True, help="QA JSONL 数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer 目录路径")
    parser.add_argument("--output_path", type=str, default="outputs/evals/qa_eval.json", help="输出 JSON 路径")
    parser.add_argument("--predictions_path", type=str, default="outputs/evals/qa_predictions.jsonl", help="预测结果 JSONL")
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
        print("[WARNING] checkpoint_path 为空，将使用随机初始化模型。QA 评估需要已训练的模型。")

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
    metrics, details = evaluate_qa(
        model, tokenizer, args.data_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    result = {
        "task": "qa_eval",
        "checkpoint": args.checkpoint_path,
        "data_path": args.data_path,
        **metrics,
    }

    write_json(args.output_path, result)
    write_jsonl(args.predictions_path, details)

    print(f"Exact Match Rate:  {metrics['exact_match_rate']:.4f}")
    print(f"Keyword Hit Rate:  {metrics['keyword_hit_rate']:.4f}")
    print(f"Avg KW Recall:     {metrics['average_keyword_recall']:.4f}")
    print(f"Non-Empty Rate:    {metrics['answer_non_empty_rate']:.4f}")
    for k, v in metrics.items():
        if k.startswith("category_"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
