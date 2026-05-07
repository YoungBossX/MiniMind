"""语言模型评估 — Validation Loss / Perplexity

对 TXT 或 JSONL 文本数据计算 token-level 评估指标。

用法:
    python evals/eval_lm.py --checkpoint_path out/pretrain_512.pth --data_path evals/data/lm_eval_sample.txt
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.core.load_model import load_model_and_tokenizer
from evals.core.metrics import compute_lm_metrics
from evals.core.io_utils import write_json, read_txt


class TextDataset(Dataset):
    """简单文本数据集：将每一行文本 tokenize 并 padding 到 max_length"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.input_ids = []
        self.loss_masks = []
        for t in texts:
            enc = tokenizer(
                t, max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            ids = enc.input_ids.squeeze(0)
            mask = (ids != tokenizer.pad_token_id).long()
            self.input_ids.append(ids)
            self.loss_masks.append(mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.loss_masks[idx]


def evaluate_lm(model, tokenizer, data_path, batch_size=4, max_length=512, device="cpu"):
    """在指定数据集上计算 validation loss 和 perplexity"""
    texts = read_txt(data_path)
    if not texts:
        raise ValueError(f"No valid text lines found in: {data_path}")

    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    total_seq_len = 0
    num_samples = 0

    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for input_ids, loss_mask in loader:
            input_ids = input_ids.to(device)
            loss_mask = loss_mask.to(device)

            # Forward: 获取 logits
            outputs = model(input_ids, attention_mask=(input_ids != tokenizer.pad_token_id).long())
            logits = outputs.logits

            # Shift: 用位置 i 预测位置 i+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = loss_mask[:, 1:].contiguous()

            # Cross entropy per token
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.size())

            # PAD 对应的 labels 设为 -100 使其被忽略
            labels_for_ce = shift_labels.clone()
            labels_for_ce[shift_mask == 0] = -100

            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels_for_ce.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

            n_tokens = shift_mask.sum().item()
            total_loss += ce.item()
            total_tokens += n_tokens
            total_seq_len += (input_ids != tokenizer.pad_token_id).sum().item()
            num_samples += input_ids.size(0)

    elapsed = time.time() - t0
    return compute_lm_metrics(
        total_loss=total_loss,
        total_tokens=int(total_tokens),
        num_samples=num_samples,
        total_sequence_length=total_seq_len,
        max_sequence_length=max_length,
        batch_size=batch_size,
        device=device,
        eval_time_seconds=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description="MiniMind Language Model Evaluation (PPL)")
    parser.add_argument("--checkpoint_path", type=str, default="", help="模型权重 .pth 路径")
    parser.add_argument("--data_path", type=str, required=True, help="TXT 评估数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer 目录路径")
    parser.add_argument("--output_path", type=str, default="outputs/evals/lm_eval.json", help="输出 JSON 路径")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not args.checkpoint_path:
        print("[WARNING] checkpoint_path 为空，将使用随机初始化模型进行评估。")

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
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    print(f"Evaluating on: {args.data_path}")
    metrics = evaluate_lm(
        model, tokenizer, args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    result = {
        "task": "lm_eval",
        "checkpoint": args.checkpoint_path,
        "data_path": args.data_path,
        **metrics,
    }

    write_json(args.output_path, result)
    print(f"Validation Loss: {metrics['validation_loss']:.4f}")
    print(f"Perplexity:     {metrics['perplexity']:.4f}")
    print(f"Evaluated Tokens: {metrics['evaluated_tokens']}")
    print(f"Time: {metrics['eval_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
