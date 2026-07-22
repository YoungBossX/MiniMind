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
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML config {config_path}: {exc}") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a top-level mapping")
    return config


def _config_section(config, name):
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"YAML config section {name!r} must be a mapping")
    return section


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


def _is_readable_file(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def _prepare_output_file(path):
    """Remove a prior result so only this invocation can satisfy the stage."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        print(f"[run_all] ERROR cannot replace stale output ({path}): {exc}")
        return False
    return True


def _prepare_output_files(*paths):
    ready = True
    for path in paths:
        ready = _prepare_output_file(path) and ready
    return ready


def _load_stage_metrics(path, returncode):
    if returncode != 0 or not _is_readable_file(path):
        return None
    try:
        metrics = read_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[run_all] ERROR invalid evaluator output ({path}): {exc}")
        return None
    if not isinstance(metrics, dict):
        print(f"[run_all] ERROR evaluator output must be a JSON object ({path})")
        return None
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="MiniMind Run All Evaluations")
    parser.add_argument("--checkpoint_path", type=str, default="", help="模型权重 .pth 路径（为空则使用随机初始化模型）")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer 目录路径")
    parser.add_argument("--config_path", type=str, default="evals/configs/eval_config.yaml", help="YAML 配置路径")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default=None, choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--skip_lm", action="store_true", help="跳过 LM 评估")
    parser.add_argument("--skip_qa", action="store_true", help="跳过 QA 评估")
    parser.add_argument("--skip_generation", action="store_true", help="跳过生成约束评估")
    parser.add_argument("--skip_speed", action="store_true", help="跳过速度评估")
    parser.add_argument(
        "--allow_random_init",
        action="store_true",
        help="仅用于 smoke test：显式允许评估随机初始化模型",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint_path and not args.allow_random_init:
        parser.error(
            "--checkpoint_path is required; pass --allow_random_init only for smoke tests"
        )
    if not args.checkpoint_path:
        print("[run_all] WARNING: checkpoint_path 为空，将使用随机初始化模型。部分评估（QA/Generation）结果无参考价值。")

    try:
        config = load_config(args.config_path)
        model_cfg = _config_section(config, "model")
        data_cfg = _config_section(config, "data")
        gen_cfg = _config_section(config, "generation")
        eval_cfg = _config_section(config, "eval")
    except ValueError as exc:
        parser.error(str(exc))

    device = args.device or model_cfg.get("device", "cpu")
    if not isinstance(device, str) or not device:
        parser.error("YAML model.device must be a non-empty string")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_size = model_cfg.get("hidden_size", 512)
    num_hidden_layers = model_cfg.get("num_hidden_layers", 8)
    output_dir = args.output_dir or eval_cfg.get("output_dir", "outputs/evals")
    dtype = args.dtype or model_cfg.get("dtype", "auto")
    if not isinstance(output_dir, str) or not output_dir:
        parser.error("YAML eval.output_dir must be a non-empty string")
    if dtype not in {"auto", "fp32", "fp16", "bf16"}:
        parser.error("YAML model.dtype must be one of auto, fp32, fp16, bf16")

    os.makedirs(output_dir, exist_ok=True)

    def _resolve_path(cfg_key, default):
        """解析配置中的数据路径（相对于项目根目录）"""
        path = data_cfg.get(cfg_key, default)
        if not isinstance(path, str) or not path:
            parser.error(f"YAML data.{cfg_key} must be a non-empty string")
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
    if model_cfg.get("use_moe", False):
        common.append("--use_moe")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_metrics = {}
    stage_status = {
        "lm": "skipped",
        "qa": "skipped",
        "generation": "skipped",
        "speed": "skipped",
    }
    failed_samples = {"qa": [], "gen": []}

    # 1. LM Evaluation
    if not args.skip_lm:
        lm_data = _resolve_path("lm_eval_path", "evals/data/lm_eval_sample.txt")
        lm_output = os.path.join(output_dir, "lm_eval.json")
        if not _prepare_output_file(lm_output):
            stage_status["lm"] = "failed"
        elif _is_readable_file(lm_data):
            lm_returncode = run_script(os.path.join(script_dir, "eval_lm.py"), common + [
                "--data_path", lm_data,
                "--output_path", lm_output,
                "--batch_size", str(eval_cfg.get("batch_size", 4)),
                "--max_length", str(eval_cfg.get("max_length", 512)),
            ])
            lm_metrics = _load_stage_metrics(lm_output, lm_returncode)
            if lm_metrics is not None:
                all_metrics["lm"] = lm_metrics
                stage_status["lm"] = "success"
            else:
                stage_status["lm"] = "failed"
        else:
            print(f"[run_all] ERROR lm_eval: data missing or unreadable ({lm_data})")
            stage_status["lm"] = "failed"

    # 2. QA Evaluation
    if not args.skip_qa:
        qa_data = _resolve_path("qa_eval_path", "evals/data/qa_eval_sample.jsonl")
        qa_output = os.path.join(output_dir, "qa_eval.json")
        qa_preds = os.path.join(output_dir, "qa_predictions.jsonl")
        if not _prepare_output_files(qa_output, qa_preds):
            stage_status["qa"] = "failed"
        elif _is_readable_file(qa_data):
            qa_returncode = run_script(os.path.join(script_dir, "eval_qa.py"), common + [
                "--data_path", qa_data,
                "--output_path", qa_output,
                "--predictions_path", qa_preds,
                "--max_new_tokens", str(gen_cfg.get("max_new_tokens", 256)),
                "--temperature", str(gen_cfg.get("temperature", 0.7)),
                "--top_p", str(gen_cfg.get("top_p", 0.9)),
            ])
            qa_metrics = _load_stage_metrics(qa_output, qa_returncode)
            if qa_metrics is not None and _is_readable_file(qa_preds):
                all_metrics["qa"] = qa_metrics
                stage_status["qa"] = "success"
            else:
                if qa_metrics is not None:
                    print(f"[run_all] ERROR qa_eval did not write predictions ({qa_preds})")
                stage_status["qa"] = "failed"
        else:
            print(f"[run_all] ERROR qa_eval: data missing or unreadable ({qa_data})")
            stage_status["qa"] = "failed"

    # 3. Generation Constraint Evaluation
    if not args.skip_generation:
        gen_data = _resolve_path("generation_eval_path", "evals/data/generation_eval_sample.jsonl")
        gen_output = os.path.join(output_dir, "generation_eval.json")
        gen_preds = os.path.join(output_dir, "generation_predictions.jsonl")
        if not _prepare_output_files(gen_output, gen_preds):
            stage_status["generation"] = "failed"
        elif _is_readable_file(gen_data):
            generation_returncode = run_script(os.path.join(script_dir, "eval_generation.py"), common + [
                "--data_path", gen_data,
                "--output_path", gen_output,
                "--predictions_path", gen_preds,
                "--max_new_tokens", str(gen_cfg.get("max_new_tokens", 256)),
                "--temperature", str(gen_cfg.get("temperature", 0.7)),
                "--top_p", str(gen_cfg.get("top_p", 0.9)),
            ])
            generation_metrics = _load_stage_metrics(
                gen_output, generation_returncode
            )
            if generation_metrics is not None and _is_readable_file(gen_preds):
                all_metrics["generation"] = generation_metrics
                stage_status["generation"] = "success"
            else:
                if generation_metrics is not None:
                    print(
                        "[run_all] ERROR generation_eval did not write "
                        f"predictions ({gen_preds})"
                    )
                stage_status["generation"] = "failed"
        else:
            print(f"[run_all] ERROR generation_eval: data missing or unreadable ({gen_data})")
            stage_status["generation"] = "failed"

    # 4. Speed Evaluation
    if not args.skip_speed:
        speed_output = os.path.join(output_dir, "speed_eval.json")
        if not _prepare_output_file(speed_output):
            stage_status["speed"] = "failed"
        else:
            speed_returncode = run_script(os.path.join(script_dir, "eval_speed.py"), common + [
                "--output_path", speed_output,
                "--max_new_tokens", str(gen_cfg.get("max_new_tokens", 256)),
                "--warmup_runs", str(eval_cfg.get("warmup_runs", 3)),
                "--repeat_runs", str(eval_cfg.get("repeat_runs", 10)),
                "--batch_size", str(eval_cfg.get("batch_size", 1)),
            ])
            speed_metrics = _load_stage_metrics(speed_output, speed_returncode)
            if speed_metrics is not None:
                all_metrics["speed"] = speed_metrics
                stage_status["speed"] = "success"
            else:
                stage_status["speed"] = "failed"

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
        stage_status=stage_status,
    )

    if "failed" in stage_status.values():
        print(f"\n[run_all] Evaluation failed. Results in: {output_dir}/")
        return 1

    print(f"\n[run_all] All requested evaluations complete. Results in: {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
