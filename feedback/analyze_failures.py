"""Turn evaluation prediction files into normalized failure records."""

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    if not path:
        return rows
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
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


def classify_qa_record(record):
    prediction = record.get("prediction", "")
    if record.get("exact_match", False):
        return None

    expected = record.get("keywords_expected", []) or []
    hit = set(record.get("keywords_hit", []) or [])
    missing = [kw for kw in expected if kw not in hit]
    if not prediction.strip():
        failure_type = "empty_answer"
    elif missing:
        failure_type = "keyword_missing"
    else:
        failure_type = "exact_mismatch"

    return {
        "source": "qa",
        "id": record.get("id", ""),
        "input": record.get("question", ""),
        "prediction": prediction,
        "reference": record.get("reference", ""),
        "failure_type": failure_type,
        "suggested_stage": "sft",
        "category": record.get("category", "unknown"),
        "details": {
            "keywords_expected": expected,
            "keywords_hit": list(hit),
            "keywords_missing": missing,
        },
    }


def classify_generation_record(record):
    if record.get("format_correct", False):
        return None

    if not record.get("json_parsed", True):
        failure_type = "json_parse_failed"
    elif not record.get("required_keys_ok", True):
        failure_type = "required_key_missing"
    elif not record.get("length_ok", True):
        failure_type = "length_violation"
    elif not record.get("must_include_ok", True):
        failure_type = "must_include_missing"
    elif not record.get("forbid_include_ok", True):
        failure_type = "forbidden_phrase_present"
    else:
        failure_type = "format_violation"

    return {
        "source": "generation",
        "id": record.get("id", ""),
        "input": record.get("prompt", ""),
        "prediction": record.get("prediction", ""),
        "reference": record.get("reference", ""),
        "failure_type": failure_type,
        "suggested_stage": "sft_format",
        "category": "generation_constraint",
        "details": {
            "constraints": record.get("constraints", {}),
            "messages": record.get("details", []),
        },
    }


def analyze_prediction_files(qa_path=None, generation_path=None):
    failures = []
    for record in read_jsonl(qa_path):
        failure = classify_qa_record(record)
        if failure:
            failures.append(failure)
    for record in read_jsonl(generation_path):
        failure = classify_generation_record(record)
        if failure:
            failures.append(failure)
    return failures


def main():
    parser = argparse.ArgumentParser(description="Analyze MiniMind evaluation failures")
    parser.add_argument("--qa_predictions", type=str, default="outputs/evals/qa_predictions.jsonl")
    parser.add_argument("--generation_predictions", type=str, default="outputs/evals/generation_predictions.jsonl")
    parser.add_argument("--output_path", type=str, default="feedback/failures.jsonl")
    args = parser.parse_args()

    failures = analyze_prediction_files(args.qa_predictions, args.generation_predictions)
    write_jsonl(args.output_path, failures)
    print(f"[analyze_failures] wrote {len(failures)} failures to {args.output_path}")


if __name__ == "__main__":
    main()
