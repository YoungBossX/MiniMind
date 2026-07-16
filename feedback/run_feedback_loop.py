"""Run the minimal eval-to-feedback loop for an evaluation output directory."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback.analyze_failures import analyze_prediction_files, write_jsonl
from feedback.build_feedback_dataset import build_sft_candidates


def run_feedback_loop(eval_dir, output_dir):
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = analyze_prediction_files(
        qa_path=eval_dir / "qa_predictions.jsonl",
        generation_path=eval_dir / "generation_predictions.jsonl",
    )
    candidates = build_sft_candidates(failures)

    failures_path = output_dir / "failures.jsonl"
    candidates_path = output_dir / "sft_candidates.jsonl"
    summary_path = output_dir / "feedback_summary.json"

    write_jsonl(failures_path, failures)
    write_jsonl(candidates_path, candidates)

    summary = {
        "eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
        "failure_count": len(failures),
        "candidate_count": len(candidates),
        "ready_candidate_count": sum(1 for item in candidates if item.get("status") == "ready"),
        "review_candidate_count": sum(1 for item in candidates if item.get("status") == "needs_review"),
        "failures_path": str(failures_path),
        "candidates_path": str(candidates_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run MiniMind eval-to-feedback loop")
    parser.add_argument("--eval_dir", type=str, default="outputs/evals")
    parser.add_argument("--output_dir", type=str, default="feedback/candidates")
    args = parser.parse_args()

    summary = run_feedback_loop(args.eval_dir, args.output_dir)
    print(
        "[run_feedback_loop] "
        f"failures={summary['failure_count']} "
        f"candidates={summary['candidate_count']} "
        f"ready={summary['ready_candidate_count']} "
        f"review={summary['review_candidate_count']}"
    )


if __name__ == "__main__":
    main()
