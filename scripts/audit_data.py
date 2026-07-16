"""Audit MiniMind training data files and registry entries."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                rows.append((line_no, {"__json_error__": str(exc)}))
    return rows


def _issue(line_no, code, message):
    return {"line": line_no, "code": code, "message": message}


def validate_dpo_conversation(conversation, side, line_no):
    issues = []
    for index, turn in enumerate(conversation):
        if not isinstance(turn, dict):
            issues.append(_issue(
                line_no,
                "dpo_invalid_turn",
                f"{side} turn {index} must be an object",
            ))
            continue
        if turn.get("role") not in {"system", "user", "assistant"}:
            issues.append(_issue(
                line_no,
                "dpo_invalid_turn",
                f"{side} turn {index} has an unsupported role",
            ))
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            issues.append(_issue(
                line_no,
                "dpo_invalid_turn",
                f"{side} turn {index} needs non-empty string content",
            ))
    return issues


def validate_sample(sample, stage, line_no):
    if "__json_error__" in sample:
        return [_issue(line_no, "invalid_json", sample["__json_error__"])]

    if stage == "pretrain":
        text = sample.get("text")
        if not isinstance(text, str) or not text.strip():
            return [_issue(line_no, "pretrain_missing_text", "pretrain sample needs non-empty text")]
        return []

    if stage in {"sft", "lora", "reason"}:
        conversations = sample.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            return [_issue(line_no, "sft_missing_conversations", "SFT sample needs conversations list")]
        roles = [turn.get("role") for turn in conversations if isinstance(turn, dict)]
        issues = []
        if "assistant" not in roles:
            issues.append(_issue(line_no, "sft_missing_assistant", "SFT sample needs an assistant turn"))
        for idx, turn in enumerate(conversations):
            content = turn.get("content") if isinstance(turn, dict) else None
            if not isinstance(content, str) or not content.strip():
                issues.append(_issue(line_no, "empty_content", f"turn {idx} has empty content"))
        return issues

    if stage == "dpo":
        issues = []
        for key in ("chosen", "rejected"):
            value = sample.get(key)
            if not isinstance(value, list) or not value:
                issues.append(_issue(line_no, f"dpo_missing_{key}", f"DPO sample needs non-empty {key} list"))
            elif not issues:
                issues.extend(validate_dpo_conversation(value, key, line_no))
        if not issues:
            if (
                sample["chosen"][-1]["role"] != "assistant"
                or sample["rejected"][-1]["role"] != "assistant"
            ):
                issues.append(_issue(
                    line_no,
                    "dpo_missing_final_assistant",
                    "chosen and rejected must both end with an assistant response",
                ))
            elif sample["chosen"][:-1] != sample["rejected"][:-1]:
                issues.append(_issue(
                    line_no,
                    "dpo_prompt_mismatch",
                    "chosen and rejected must share the same conversation prefix",
                ))
        return issues

    if stage in {"rlaif", "ppo", "grpo"}:
        prompt = sample.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return []

        conversations = sample.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            return [_issue(
                line_no,
                "rlaif_missing_conversations",
                "RLAIF sample needs prompt or at least one user/assistant conversation pair",
            )]
        issues = []
        for idx, turn in enumerate(conversations):
            content = turn.get("content") if isinstance(turn, dict) else None
            if not isinstance(content, str) or not content.strip():
                issues.append(_issue(line_no, "empty_content", f"turn {idx} has empty content"))
        return issues

    return [_issue(line_no, "unknown_stage", f"unsupported stage: {stage}")]


def audit_jsonl_file(path, stage):
    path = Path(path)
    rows = load_jsonl(path)
    issues = []
    valid_samples = 0

    for line_no, sample in rows:
        sample_issues = validate_sample(sample, stage, line_no)
        if sample_issues:
            issues.extend(sample_issues)
        else:
            valid_samples += 1

    return {
        "path": str(path),
        "stage": stage,
        "sha256": sha256_file(path) if path.exists() else "",
        "total_samples": len(rows),
        "valid_samples": valid_samples,
        "invalid_samples": len(rows) - valid_samples,
        "issues": issues,
    }


def _resolve_data_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_registry(registry_dir):
    registry_dir = Path(registry_dir)
    entries = []
    for path in sorted(list(registry_dir.glob("*.yaml")) + list(registry_dir.glob("*.yml"))):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in data.get("datasets", []):
            entry = dict(item)
            entry["registry_file"] = str(path)
            entries.append(entry)
    return entries


def audit_registry(registry_dir):
    reports = []
    for entry in load_registry(registry_dir):
        data_path = _resolve_data_path(entry["path"])
        if not data_path.exists():
            reports.append({
                "path": str(data_path),
                "stage": entry.get("stage", "unknown"),
                "sha256": "",
                "total_samples": 0,
                "valid_samples": 0,
                "invalid_samples": 0,
                "issues": [_issue(0, "missing_file", "registered data file does not exist")],
            })
            continue
        reports.append(audit_jsonl_file(data_path, entry["stage"]))
    return reports


def main():
    parser = argparse.ArgumentParser(description="Audit MiniMind data files")
    parser.add_argument("--data_path", type=str, default="", help="Single JSONL file to audit")
    parser.add_argument("--stage", type=str, default="", help="Dataset stage for --data_path")
    parser.add_argument("--registry_dir", type=str, default="data_registry", help="Registry directory")
    parser.add_argument("--output_path", type=str, default="outputs/data_audit.json")
    args = parser.parse_args()

    if args.data_path:
        if not args.stage:
            raise ValueError("--stage is required with --data_path")
        result = [audit_jsonl_file(args.data_path, args.stage)]
    else:
        result = audit_registry(PROJECT_ROOT / args.registry_dir)

    output_path = _resolve_data_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit_data] report saved: {output_path}")


if __name__ == "__main__":
    main()
