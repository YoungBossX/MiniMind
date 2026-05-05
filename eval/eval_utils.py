"""MiniMind 评测系统 — 共享工具模块"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import subprocess
from datetime import datetime, timezone
import torch
import numpy as np


def get_git_commit():
    """获取当前 HEAD 的短 commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_report(stage, metrics, assertions, output_dir):
    """生成并保存 JSON + MD 报告，返回结果字典"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    commit = get_git_commit()
    passed = all(a["passed"] for a in assertions)

    report = {
        "stage": stage,
        "timestamp": timestamp,
        "git_commit": commit,
        "passed": passed,
        "metrics": {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()},
        "assertions": assertions,
    }

    base = os.path.join(output_dir, f"{stage}_{timestamp}")
    save_json_report(report, base + ".json")
    save_md_report(report, base + ".md")

    return report


def save_json_report(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"[Report] JSON saved: {path}")


def save_md_report(report, path):
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
        icon = "PASS" if a["passed"] else "FAIL"
        lines.append(f"| {a['name']} | {icon} | {a.get('detail', '')} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] MD saved: {path}")
