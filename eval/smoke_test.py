"""MiniMind 评测系统 — 框架正确性 Smoke Test

每个训练管线快速跑 50 步验证框架完整性：
  - 模型初始化 → forward 无误
  - 梯度正常流动（非零、非 NaN）
  - loss 下降 > 10%
  - checkpoint 存取一致
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings
import torch

from model.MiniMindModel import MiniMindConfig, MiniMindForCausalLM
from eval.eval_utils import (
    generate_report, check_grad_flow, verify_checkpoint_roundtrip,
    init_swanlab, log_to_swanlab, make_small_config,
)

warnings.filterwarnings("ignore")

SMOKE_STEPS = 50
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def assertion(name, passed, detail=""):
    return {"name": name, "passed": passed, "detail": detail}


def run_stage(stage_name, config, metrics, assertions, use_swanlab=False):
    """输出终端结果，生成报告，可选 SwanLab 上报"""
    report = generate_report(stage_name, metrics, assertions, REPORT_DIR)

    status = "PASS" if report["passed"] else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {stage_name}: {status}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"  assertions: {len([a for a in assertions if a['passed']])}/{len(assertions)} passed")

    if use_swanlab:
        log_to_swanlab(stage_name, metrics)

    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description="MiniMind Smoke Test")
    parser.add_argument("--all", action="store_true", help="运行所有管线 smoke test")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["pretrain", "sft", "lora", "dpo", "reason", "ppo", "grpo"])
    parser.add_argument("--skip-rl", action="store_true", help="跳过 PPO/GRPO (需要 Reward Model)")
    parser.add_argument("--use-wandb", action="store_true", help="上报到 SwanLab")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.use_wandb:
        init_swanlab("MiniMind-Eval")

    os.makedirs(REPORT_DIR, exist_ok=True)

    if not args.all and not args.stage:
        parser.print_help()
        print("\n请指定 --all 或 --stage STAGE")
        return

    all_stages = ["pretrain", "sft", "lora", "dpo", "reason", "ppo", "grpo"]
    if args.stage:
        stages = [args.stage]
    else:
        stages = all_stages
    if args.skip_rl:
        stages = [s for s in stages if s not in ("ppo", "grpo")]

    results = {}
    for stage in stages:
        fn = globals().get(f"smoke_{stage}")
        if fn:
            print(f"\n{'#'*60}\n#  Running smoke test: {stage}\n{'#'*60}")
            results[stage] = fn(args.device, args.use_wandb)
        else:
            print(f"[WARN] Unknown stage: {stage}")

    all_passed = all(results.values()) if results else False
    print(f"\n{'='*60}")
    print(f"  OVERALL: {'PASS' if all_passed else 'FAIL'} ({sum(results.values())}/{len(results)} passed)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
