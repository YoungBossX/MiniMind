"""MiniMind 评测系统 — 共享工具模块"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import subprocess
from datetime import datetime, timezone
import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_git_commit():
    """获取当前 HEAD 的短 commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _to_serializable(v):
    """安全转换 metric 值为可序列化类型"""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        if v.numel() == 1:
            return v.item()
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    return v


def generate_report(stage, metrics, assertions, output_dir):
    """生成并保存 JSON + MD 报告，返回结果字典"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    commit = get_git_commit()
    passed = all(a.get("passed", False) for a in assertions)

    report = {
        "stage": stage,
        "timestamp": timestamp,
        "git_commit": commit,
        "passed": passed,
        "metrics": {k: _to_serializable(v) for k, v in metrics.items()},
        "assertions": assertions,
    }

    base = os.path.join(output_dir, f"{stage}_{timestamp}")
    save_json_report(report, base + ".json")
    save_md_report(report, base + ".md")

    return report


def save_json_report(report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Report] JSON saved: {path}")


def save_md_report(report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"# MiniMind Eval — {report['stage']}",
        "",
        f"**Time:** {report['timestamp']}  ",
        f"**Commit:** `{report['git_commit']}`  ",
        f"**Passed:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in report["metrics"].items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Assertions",
        "",
        "| Assertion | Result | Detail |",
        "|-----------|--------|--------|",
    ]
    for a in report["assertions"]:
        icon = "PASS" if a.get("passed", False) else "FAIL"
        lines.append(f"| {a.get('name', '?')} | {icon} | {a.get('detail', '')} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] MD saved: {path}")


_swanlab_run = None


def init_swanlab(project="MiniMind-Eval", run_name=None):
    """初始化 SwanLab（仅在主进程调用）"""
    global _swanlab_run
    try:
        import swanlab
        _swanlab_run = swanlab.init(project=project, name=run_name)
        print(f"[SwanLab] Initialized: {project}")
    except ImportError:
        _swanlab_run = None
        print("[SwanLab] swanlab not installed, skipping")


def log_to_swanlab(stage, metrics, step=0):
    """上报指标到 SwanLab，命名空间为 eval/{stage}/{metric}"""
    if _swanlab_run is None:
        return
    import swanlab
    data = {f"eval/{stage}/{k}": _to_serializable(v) for k, v in metrics.items()}
    swanlab.log(data, step=step)


def check_grad_flow(model):
    """检查模型梯度：返回 {grad_norm, has_nan, has_grad}"""
    total_norm = 0.0
    has_nan = False
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
            if not has_nan and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan = True
    return {
        "grad_norm": total_norm ** 0.5,
        "has_nan": has_nan,
        "has_grad": has_grad,
    }


def verify_checkpoint_roundtrip(model, save_path, sample_input, device="cpu"):
    """保存 checkpoint → 加载 → 比较 forward 输出。返回 (allclose, detail)"""
    import tempfile
    import os as _os

    model.eval()
    with torch.no_grad():
        orig_output = model(sample_input)
        if hasattr(orig_output, "logits"):
            orig = orig_output.logits.detach().clone()
        else:
            orig = orig_output.detach().clone()

    fd, tmp = tempfile.mkstemp(suffix=".pth", prefix="_eval_ckpt_")
    _os.close(fd)
    try:
        torch.save(model.state_dict(), tmp)

        from model.MiniMindModel import MiniMindConfig, MiniMindForCausalLM
        reloaded_model = MiniMindForCausalLM(model.config)
        reloaded_model.load_state_dict(torch.load(tmp, map_location=device), strict=True)
        reloaded_model.to(device).eval()

        with torch.no_grad():
            reloaded_output = reloaded_model(sample_input)
            if hasattr(reloaded_output, "logits"):
                reloaded_logits = reloaded_output.logits.detach()
            else:
                reloaded_logits = reloaded_output.detach()

        allclose = torch.allclose(orig.float(), reloaded_logits.float(), rtol=1e-3, atol=1e-5)
        max_diff = (orig.float() - reloaded_logits.float()).abs().max().item()
        return allclose, f"max_diff={max_diff:.2e}"
    finally:
        if _os.path.exists(tmp):
            _os.remove(tmp)


def make_small_config(use_moe=False):
    """创建 MiniMindConfig 最小实例用于 smoke test"""
    from model.MiniMindModel import MiniMindConfig
    return MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        use_moe=use_moe,
        max_position_embeddings=512,
        dropout=0.0,
    )
