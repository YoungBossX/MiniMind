"""Compare two evaluation output directories and report metric deltas."""

import argparse
import json
from pathlib import Path


EVAL_FILES = {
    "lm": "lm_eval.json",
    "qa": "qa_eval.json",
    "generation": "generation_eval.json",
    "speed": "speed_eval.json",
}


LOWER_IS_BETTER_HINTS = (
    "loss",
    "perplexity",
    "latency",
    "memory",
    "time",
)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def metric_direction(name):
    lowered = name.lower()
    if any(hint in lowered for hint in LOWER_IS_BETTER_HINTS):
        return "lower"
    return "higher"


def metric_status(name, delta):
    if delta == 0:
        return "unchanged"
    direction = metric_direction(name)
    if direction == "higher":
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def collect_metrics(eval_dir):
    eval_dir = Path(eval_dir)
    metrics = {}
    for prefix, filename in EVAL_FILES.items():
        data = read_json(eval_dir / filename)
        for key, value in data.items():
            if is_numeric(value):
                metrics[f"{prefix}.{key}"] = float(value)
    return metrics


def compare_eval_dirs(baseline_dir, current_dir):
    baseline = collect_metrics(baseline_dir)
    current = collect_metrics(current_dir)
    metric_names = sorted(set(baseline) | set(current))
    comparisons = {}
    for name in metric_names:
        base_value = baseline.get(name)
        current_value = current.get(name)
        if base_value is None or current_value is None:
            status = "missing"
            delta = None
        else:
            delta = round(current_value - base_value, 12)
            status = metric_status(name, delta)
        comparisons[name] = {
            "baseline": base_value,
            "current": current_value,
            "delta": delta,
            "direction": metric_direction(name),
            "status": status,
        }
    return {
        "baseline_dir": str(baseline_dir),
        "current_dir": str(current_dir),
        "metrics": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two MiniMind eval output directories")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--current_dir", required=True)
    parser.add_argument("--output_path", default="outputs/evals/compare_runs.json")
    args = parser.parse_args()

    comparison = compare_eval_dirs(args.baseline_dir, args.current_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compare_runs] report saved: {output_path}")


if __name__ == "__main__":
    main()
