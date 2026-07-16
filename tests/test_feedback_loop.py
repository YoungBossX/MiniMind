import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / ".tmp" / "feedback_loop_tests"


def make_case_dir(name):
    path = TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_data_reports_counts_hash_and_schema_issues():
    from scripts.audit_data import audit_jsonl_file

    tmp_path = make_case_dir("audit")
    data_path = tmp_path / "sft.jsonl"
    write_jsonl(
        data_path,
        [
            {
                "conversations": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好，我是 MiniMind。"},
                ]
            },
            {"conversations": [{"role": "user", "content": ""}]},
        ],
    )

    report = audit_jsonl_file(data_path, stage="sft")

    assert report["path"] == str(data_path)
    assert report["stage"] == "sft"
    assert report["total_samples"] == 2
    assert report["valid_samples"] == 1
    assert report["sha256"]
    assert report["issues"][0]["code"] == "sft_missing_assistant"


def test_audit_data_accepts_rlaif_conversation_pairs():
    from scripts.audit_data import audit_jsonl_file

    tmp_path = make_case_dir("rlaif_audit")
    data_path = tmp_path / "rlaif.jsonl"
    write_jsonl(
        data_path,
        [
            {
                "conversations": [
                    {"role": "user", "content": "给出一个安全建议"},
                    {"role": "assistant", "content": "先确认环境风险，再采取行动。"},
                ]
            }
        ],
    )

    report = audit_jsonl_file(data_path, stage="rlaif")

    assert report["total_samples"] == 1
    assert report["valid_samples"] == 1
    assert report["issues"] == []


def test_audit_data_rejects_dpo_pairs_with_different_prompts():
    from scripts.audit_data import validate_sample

    sample = {
        "chosen": [
            {"role": "user", "content": "Explain recursion."},
            {"role": "assistant", "content": "A clear explanation."},
        ],
        "rejected": [
            {"role": "user", "content": "Explain iteration."},
            {"role": "assistant", "content": "An unrelated answer."},
        ],
    }

    issue_codes = {issue["code"] for issue in validate_sample(sample, "dpo", 1)}

    assert "dpo_prompt_mismatch" in issue_codes


def test_audit_data_requires_dpo_responses_to_end_with_assistant():
    from scripts.audit_data import validate_sample

    conversation = [{"role": "user", "content": "Explain recursion."}]
    issue_codes = {
        issue["code"]
        for issue in validate_sample(
            {"chosen": conversation, "rejected": conversation}, "dpo", 1
        )
    }

    assert "dpo_missing_final_assistant" in issue_codes


def test_audit_data_rejects_invalid_dpo_turns():
    from scripts.audit_data import validate_sample

    conversation = [
        {"role": "tool", "content": "not a supported role"},
        {"role": "assistant", "content": "A response."},
    ]
    issue_codes = {
        issue["code"]
        for issue in validate_sample(
            {"chosen": conversation, "rejected": conversation}, "dpo", 1
        )
    }

    assert "dpo_invalid_turn" in issue_codes


def test_audit_data_rejects_dpo_turns_with_empty_content():
    from scripts.audit_data import validate_sample

    conversation = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": ""},
    ]
    issue_codes = {
        issue["code"]
        for issue in validate_sample(
            {"chosen": conversation, "rejected": conversation}, "dpo", 1
        )
    }

    assert "dpo_invalid_turn" in issue_codes


def test_audit_data_rejects_non_object_dpo_turns():
    from scripts.audit_data import validate_sample

    conversation = [
        {"role": "user", "content": "Question"},
        "invalid turn",
    ]
    issue_codes = {
        issue["code"]
        for issue in validate_sample(
            {"chosen": conversation, "rejected": conversation}, "dpo", 1
        )
    }

    assert "dpo_invalid_turn" in issue_codes


def test_analyze_failures_classifies_qa_and_generation_predictions():
    from feedback.analyze_failures import analyze_prediction_files

    tmp_path = make_case_dir("analyze")
    qa_path = tmp_path / "qa_predictions.jsonl"
    gen_path = tmp_path / "generation_predictions.jsonl"
    write_jsonl(
        qa_path,
        [
            {
                "id": "qa-1",
                "question": "法国首都是哪里？",
                "prediction": "法国的首都是里昂。",
                "reference": "法国首都是巴黎。",
                "exact_match": False,
                "keywords_expected": ["巴黎", "法国"],
                "keywords_hit": ["法国"],
                "category": "geo",
            }
        ],
    )
    write_jsonl(
        gen_path,
        [
            {
                "id": "gen-1",
                "prompt": "返回 JSON",
                "prediction": "name: minimind",
                "format_correct": False,
                "json_parsed": False,
                "required_keys_ok": False,
                "length_ok": True,
                "must_include_ok": True,
                "forbid_include_ok": True,
                "details": ["failed to parse JSON from output", "missing required key: 'name'"],
            }
        ],
    )

    failures = analyze_prediction_files(qa_path=qa_path, generation_path=gen_path)

    assert [f["source"] for f in failures] == ["qa", "generation"]
    assert failures[0]["failure_type"] == "keyword_missing"
    assert failures[0]["suggested_stage"] == "sft"
    assert failures[1]["failure_type"] == "json_parse_failed"
    assert failures[1]["suggested_stage"] == "sft_format"


def test_build_feedback_dataset_creates_ready_and_review_candidates():
    from feedback.build_feedback_dataset import build_sft_candidates

    failures = [
        {
            "source": "qa",
            "id": "qa-1",
            "input": "法国首都是哪里？",
            "prediction": "里昂",
            "reference": "法国首都是巴黎。",
            "failure_type": "keyword_missing",
            "suggested_stage": "sft",
        },
        {
            "source": "generation",
            "id": "gen-1",
            "input": "返回 JSON",
            "prediction": "name: minimind",
            "reference": "",
            "failure_type": "json_parse_failed",
            "suggested_stage": "sft_format",
        },
    ]

    candidates = build_sft_candidates(failures)

    assert candidates[0]["status"] == "ready"
    assert candidates[0]["conversations"][0]["role"] == "user"
    assert candidates[0]["conversations"][1]["content"] == "法国首都是巴黎。"
    assert candidates[1]["status"] == "needs_review"
    assert candidates[1]["metadata"]["failure_type"] == "json_parse_failed"


def test_compare_runs_reports_metric_deltas():
    from evals.compare_runs import compare_eval_dirs

    tmp_path = make_case_dir("compare")
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    baseline.mkdir()
    current.mkdir()
    (baseline / "qa_eval.json").write_text(
        json.dumps({"exact_match_rate": 0.2, "average_keyword_recall": 0.4}),
        encoding="utf-8",
    )
    (current / "qa_eval.json").write_text(
        json.dumps({"exact_match_rate": 0.5, "average_keyword_recall": 0.3}),
        encoding="utf-8",
    )

    comparison = compare_eval_dirs(baseline, current)

    assert comparison["metrics"]["qa.exact_match_rate"]["delta"] == 0.3
    assert comparison["metrics"]["qa.exact_match_rate"]["status"] == "improved"
    assert comparison["metrics"]["qa.average_keyword_recall"]["status"] == "regressed"


def test_run_feedback_loop_writes_failures_and_candidates():
    from feedback.run_feedback_loop import run_feedback_loop

    tmp_path = make_case_dir("run_loop")
    eval_dir = tmp_path / "evals"
    output_dir = tmp_path / "feedback"
    write_jsonl(
        eval_dir / "qa_predictions.jsonl",
        [
            {
                "id": "qa-1",
                "question": "2+2 等于几？",
                "prediction": "5",
                "reference": "4",
                "exact_match": False,
                "keywords_expected": ["4"],
                "keywords_hit": [],
                "category": "math",
            }
        ],
    )

    summary = run_feedback_loop(eval_dir, output_dir)

    assert summary["failure_count"] == 1
    assert summary["candidate_count"] == 1
    assert (output_dir / "failures.jsonl").exists()
    assert (output_dir / "sft_candidates.jsonl").exists()
