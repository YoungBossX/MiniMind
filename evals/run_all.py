"""一键运行全部评估并汇总报告

依次运行 lm_eval → qa_eval → generation_eval → speed_eval，
汇总结果生成 Markdown 报告。

用法:
    python evals/run_all.py --checkpoint_path out/dpo_512.pth
    python evals/run_all.py --checkpoint_path out/dpo_512.pth --config_path evals/configs/eval_config.yaml
"""

import os
import sys
import argparse
import subprocess
import json
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.core.report import generate_markdown_report
from evals.core.io_utils import read_json


def load_config(config_path):
    """加载 YAML 配置文件"""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_script(script_path, args_list):
    """运行子评估脚本并返回结果"""
    cmd = [sys.executable, script_path] + args_list
    print(f"\n{'='*60}")
    print(f"[run_all] Executing: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[run_all] WARNING: {os.path.basename(script_path)} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="MiniMind Run All Evaluations")
    parser.add_argument("--checkpoint_path", type=str, default="", help="模型权重 .pth 路径（为空则使用随机初始化模型）")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer 目录路径")
    parser.add_argument("--config_path", type=str, default="evals/configs/eval_config.yaml", help="YAML 配置路径")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/evals", help="输出目录")
    parser.add_argument("--skip_lm", action="store_true", help="跳过 LM 评估")
    parser.add_argument("--skip_qa", action="store_true", help="跳过 QA 评估")
    parser.add_argument("--skip_generation", action="store_true", help="跳过生成约束评估")
    parser.add_argument("--skip_speed", action="store_true", help="跳过速度评估")
    args = parser.parse_args()

    if not args.checkpoint_path:
        print("[run_all] WARNING: checkpoint_path 为空，将使用随机初始化模型。部分评估（QA/Generation）结果无参考价值。")

    config = load_config(args.config_path)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    gen_cfg = config.get("generation", {})
    eval_cfg = config.get("eval", {})

    device = args.device or model_cfg.get("device", "cpu")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_size = model_cfg.get("hidden_size", 512)
    num_hidden_layers = model_cfg.get("num_hidden_layers", 8)
    output_dir = args.output_dir or eval_cfg.get("output_dir", "outputs/evals")
    dtype = args.dtype or model_cfg.get("dtype", "auto")

    os.makedirs(output_dir, exist_ok=True)

    def _resolve_path(cfg_key, default):
        """解析配置中的数据路径（相对于项目根目录）"""
        path = data_cfg.get(cfg_key, default)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
        return path

    # 公共参数
    common = [
        "--checkpoint_path", args.checkpoint_path,
        "--tokenizer_path", args.tokenizer_path or "",
        "--hidden_size", str(hidden_size),
        "--num_hidden_layers", str(num_hidden_layers),
        "--device", device,
        "--dtype", dtype,
        "--seed", str(args.seed),
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_metrics = {}
    failed_samples = {"qa": [], "gen": []}

    # 1. LM Evaluation
    if not args.skip_lm:
        lm_data = _resolve_path("lm_eval_path", "evals/data/lm_eval_sample.txt")
        lm_output = os.path.join(output_dir, "lm_eval.json")
        if os.path.exists(lm_data):
            run_script(os.path.join(script_dir, "eval_lm.py"), common + [
                "--data_path", lm_data,
                "--output_path", lm_output,
                "--batch_size", str(eval_cfg.get("batch_size", 4)),
                "--max_length", str(eval_cfg.get("max_length", 512)),
            ])
            if os.path.exists(lm_output):
                all_metrics["lm"] = read_json(lm_output)
        else:
            print(f"[run_all] SKIP lm_eval: data not found ({lm_data})")

    # 2. QA Evaluation
    if not args.skip_qa:
        qa_data = _resolve_path("qa_eval_path", "evals/data/qa_eval_sample.jsonl")
        qa_output = os.path.join(output_dir, "qa_eval.json")
        qa_preds = os.path.join(output_dir, "qa_predictions.jsonl")
        if os.path.exists(qa_data):
            run_script(os.path.join(script_dir, "eval_qa.py"), common + [
                "--data_path", qa_data,
                "--output_path", qa_output,
                "--predictions_path", qa_preds,
                "--max_new_tokens", str(gen_cfg.get("max_new_tokens", 256)),
                "--temperature", str(gen_cfg.get("temperature", 0.7)),
                "--top_p", str(gen_cfg.get("top_p", 0.9)),
            ])
            if os.path.exists(qa_output):
                all_metrics["qa"] = read_json(qa_output)
        else:
            print(f"[run_all] SKIP qa_eval: data not found ({qa_data})")

    # 3. Generation Constraint Evaluation
    if not args.skip_generation:
        gen_data = _resolve_path("generation_eval_path", "evals/data/generation_eval_sample.jsonl")
        gen_output = os.path.join(output_dir, "generation_eval.json")
        gen_preds = os.path.join(output_dir, "generation_predictions.jsonl")
        if os.path.exists(gen_data):
            run_script(os.path.join(script_dir, "eval_generation.py"), common + [
                "--data_path", gen_data,
                "--output_path", gen_output,
                "--predictions_path", gen_preds,
                "--max_new_tokens", str(gen_cfg.get("max_new_tokens", 256)),
                "--temperature", str(gen_cfg.get("temperature", 0.7)),
                "--top_p", str(gen_cfg.get("top_p", 0.9)),
            ])
            if os.path.exists(gen_output):
                all_metrics["generation"] = read_json(gen_output)
        else:
            print(f"[run_all] SKIP generation_eval: data not found ({gen_data})")

    # 4. Speed Evaluation
    if not args.skip_speed:
        speed_output = os.path.join(output_dir, "speed_eval.json")
        run_script(os.path.join(script_dir, "eval_speed.py"), common + [
            "--output_path", speed_output,
            "--max_new_tokens", str(gen_cfg.get("max_new_tokens", 256)),
            "--warmup_runs", str(eval_cfg.get("warmup_runs", 3)),
            "--repeat_runs", str(eval_cfg.get("repeat_runs", 10)),
        ])
        if os.path.exists(speed_output):
            all_metrics["speed"] = read_json(speed_output)

    # Generate report
    print(f"\n{'='*60}")
    print("[run_all] Generating summary report...")
    print(f"{'='*60}")

    generate_markdown_report(
        checkpoint_path=args.checkpoint_path,
        all_metrics=all_metrics,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
        seed=args.seed,
        failed_samples=failed_samples,
    )

    print(f"\n[run_all] All evaluations complete. Results in: {output_dir}/")


if __name__ == "__main__":
    main()
