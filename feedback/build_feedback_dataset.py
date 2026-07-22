"""Build human-reviewable SFT candidates from failure records."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sft_candidates(failures):
    candidates = []
    for failure in failures:
        prompt = failure.get("input", "")
        reference = failure.get("reference", "")
        status = "needs_review"
        assistant_content = ""
        candidates.append({
            "status": status,
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "source": failure.get("source", ""),
                "source_id": failure.get("id", ""),
                "failure_type": failure.get("failure_type", ""),
                "suggested_stage": failure.get("suggested_stage", "sft"),
                "prediction": failure.get("prediction", ""),
                "reference_suggestion": reference,
                "details": failure.get("details", {}),
            },
        })
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Build feedback SFT candidates")
    parser.add_argument("--failures_path", type=str, default="feedback/failures.jsonl")
    parser.add_argument("--output_path", type=str, default="feedback/candidates/sft_candidates.jsonl")
    args = parser.parse_args()

    failures = read_jsonl(args.failures_path)
    candidates = build_sft_candidates(failures)
    write_jsonl(args.output_path, candidates)
    print(f"[build_feedback_dataset] wrote {len(candidates)} candidates to {args.output_path}")


if __name__ == "__main__":
    main()
