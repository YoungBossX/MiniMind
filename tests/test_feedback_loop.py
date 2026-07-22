import json

import pytest


def make_case_dir(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_data_reports_counts_hash_and_schema_issues(tmp_path):
    from scripts.audit_data import audit_jsonl_file

    tmp_path = make_case_dir(tmp_path, "audit")
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


def test_audit_data_accepts_rlaif_conversation_pairs(tmp_path):
    from scripts.audit_data import audit_jsonl_file

    tmp_path = make_case_dir(tmp_path, "rlaif_audit")
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


def test_audit_data_reports_exact_duplicate_rows_once(tmp_path):
    from scripts.audit_data import audit_jsonl_file

    tmp_path = make_case_dir(tmp_path, "duplicate_audit")
    data_path = tmp_path / "sft.jsonl"
    row = {
        "conversations": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    write_jsonl(data_path, [row, row])

    report = audit_jsonl_file(data_path, stage="sft")

    assert report["total_samples"] == 2
    assert report["valid_samples"] == 1
    assert report["duplicate_samples"] == 1
    assert [issue["code"] for issue in report["issues"]] == ["duplicate_row"]


def test_audit_registry_missing_file_preserves_report_schema(tmp_path):
    from scripts.audit_data import audit_registry

    tmp_path = make_case_dir(tmp_path, "missing_registry_file")
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        "datasets:\n  - path: missing.jsonl\n    stage: pretrain\n",
        encoding="utf-8",
    )

    report = audit_registry(tmp_path)[0]

    assert report["duplicate_samples"] == 0


def test_audit_data_reports_non_object_json_rows_without_crashing(tmp_path):
    from scripts.audit_data import audit_jsonl_file

    tmp_path = make_case_dir(tmp_path, "non_object_audit")
    data_path = tmp_path / "pretrain.jsonl"
    data_path.write_text('[]\n42\n"text"\n', encoding="utf-8")

    report = audit_jsonl_file(data_path, stage="pretrain")

    assert report["total_samples"] == 3
    assert report["valid_samples"] == 0
    assert report["invalid_samples"] == 3
    assert [issue["code"] for issue in report["issues"]] == [
        "invalid_sample_type",
        "invalid_sample_type",
        "invalid_sample_type",
    ]


@pytest.mark.parametrize("stage", ["sft", "lora", "reason"])
def test_audit_data_rejects_unsupported_sft_roles(stage):
    from scripts.audit_data import validate_sample

    sample = {
        "conversations": [
            {"role": "tool", "content": "Tool output"},
            {"role": "assistant", "content": "A response."},
        ]
    }

    issues = validate_sample(sample, stage, 1)

    assert [issue["code"] for issue in issues] == ["sft_invalid_role"]


@pytest.mark.parametrize("stage", ["sft", "lora", "reason"])
def test_audit_data_rejects_non_object_sft_turns(stage):
    from scripts.audit_data import validate_sample

    sample = {
        "conversations": [
            {"role": "user", "content": "Question"},
            "invalid turn",
            {"role": "assistant", "content": "A response."},
        ]
    }

    issues = validate_sample(sample, stage, 1)

    assert [issue["code"] for issue in issues] == ["sft_invalid_turn"]


@pytest.mark.parametrize("stage", ["sft", "lora", "reason"])
@pytest.mark.parametrize("content", ["", "   ", None, 123])
def test_audit_data_rejects_empty_or_non_string_sft_content(stage, content):
    from scripts.audit_data import validate_sample

    sample = {
        "conversations": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "A response."},
        ]
    }

    issues = validate_sample(sample, stage, 1)

    assert [issue["code"] for issue in issues] == ["empty_content"]


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


def test_audit_data_rejects_dpo_pairs_with_identical_responses():
    from scripts.audit_data import validate_sample

    conversation = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Same answer"},
    ]
    issue_codes = {
        issue["code"]
        for issue in validate_sample(
            {"chosen": conversation, "rejected": conversation}, "dpo", 1
        )
    }

    assert "dpo_identical_responses" in issue_codes


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


def test_audit_data_validates_chosen_and_rejected_independently():
    from scripts.audit_data import validate_sample

    sample = {
        "chosen": [
            {"role": "tool", "content": "Unsupported role"},
            {"role": "assistant", "content": "Chosen response."},
        ],
        "rejected": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Rejected response."},
        ],
    }

    issues = validate_sample(sample, "dpo", 1)

    assert [(issue["code"], issue["message"]) for issue in issues] == [
        ("dpo_invalid_turn", "chosen turn 0 has an unsupported role"),
        ("dpo_invalid_turn", "rejected turn 0 needs non-empty string content"),
    ]


def test_audit_data_validates_a_bad_branch_when_the_other_branch_is_missing():
    from scripts.audit_data import validate_sample

    issues = validate_sample(
        {"chosen": [], "rejected": [None]}, "dpo", 1
    )

    issue_codes = [issue["code"] for issue in issues]
    assert issue_codes == ["dpo_missing_chosen", "dpo_invalid_turn"]
    assert not {
        "dpo_missing_final_assistant",
        "dpo_prompt_mismatch",
        "dpo_identical_responses",
    }.intersection(issue_codes)


def test_analyze_failures_classifies_qa_and_generation_predictions(tmp_path):
    from feedback.analyze_failures import analyze_prediction_files

    tmp_path = make_case_dir(tmp_path, "analyze")
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


def test_build_feedback_dataset_keeps_evaluation_references_review_only():
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

    assert candidates[0]["status"] == "needs_review"
    assert candidates[0]["conversations"][0]["role"] == "user"
    assert candidates[0]["conversations"][1]["content"] == ""
    assert candidates[0]["metadata"]["reference_suggestion"] == "法国首都是巴黎。"
    assert candidates[1]["status"] == "needs_review"
    assert candidates[1]["metadata"]["failure_type"] == "json_parse_failed"


def test_compare_runs_reports_metric_deltas(tmp_path):
    from evals.compare_runs import compare_eval_dirs

    tmp_path = make_case_dir(tmp_path, "compare")
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


def test_run_feedback_loop_writes_failures_and_candidates(tmp_path):
    from feedback.run_feedback_loop import run_feedback_loop

    tmp_path = make_case_dir(tmp_path, "run_loop")
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
