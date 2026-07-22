import sys
from pathlib import Path

import pytest

from evals import run_all
from evals.core.metrics import compute_speed_metrics


def test_run_all_returns_nonzero_when_a_requested_evaluator_fails(monkeypatch, tmp_path):
    reported = {}
    monkeypatch.setattr(run_all, "run_script", lambda *_: 1)
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all.py",
            "--allow_random_init",
            "--output_dir",
            str(tmp_path),
            "--skip_qa",
            "--skip_generation",
            "--skip_speed",
        ],
    )

    assert run_all.main() == 1
    assert reported["stage_status"]["lm"] == "failed"


def test_run_all_forwards_configured_batch_size_to_speed_evaluator(monkeypatch, tmp_path):
    commands = []

    def fake_run_script(_script_path, args_list):
        commands.append(args_list)
        output_path = Path(args_list[args_list.index("--output_path") + 1])
        output_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_all, "run_script", fake_run_script)
    monkeypatch.setattr(run_all, "load_config", lambda _: {"eval": {"batch_size": 7}})

    assert run_all.main([
        "--allow_random_init",
        "--output_dir", str(tmp_path),
        "--skip_lm", "--skip_qa", "--skip_generation",
    ]) == 0

    speed_args = commands[0]
    assert speed_args[speed_args.index("--batch_size") + 1] == "7"


def test_speed_metrics_record_batch_size_provenance():
    metrics = compute_speed_metrics(
        latencies_ms=[10.0],
        total_generated_tokens=8,
        total_prompt_tokens=4,
        batch_size=4,
    )

    assert metrics["batch_size"] == 4


def test_run_all_requires_explicit_random_init_smoke_mode(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        run_all.main([
            "--output_dir", str(tmp_path),
            "--skip_lm", "--skip_qa", "--skip_generation", "--skip_speed",
        ])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("stage", "data_key", "skip_args"),
    [
        ("lm", "lm_eval_path", ["--skip_qa", "--skip_generation", "--skip_speed"]),
        ("qa", "qa_eval_path", ["--skip_lm", "--skip_generation", "--skip_speed"]),
        (
            "generation",
            "generation_eval_path",
            ["--skip_lm", "--skip_qa", "--skip_speed"],
        ),
    ],
)
def test_run_all_marks_missing_requested_input_as_failed(
    monkeypatch, tmp_path, stage, data_key, skip_args
):
    missing_data = tmp_path / f"missing-{stage}.jsonl"
    reported = {}

    monkeypatch.setattr(
        run_all,
        "load_config",
        lambda _: {"data": {data_key: str(missing_data)}},
    )
    monkeypatch.setattr(
        run_all,
        "run_script",
        lambda *_: pytest.fail("an evaluator must not run with missing input"),
    )
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )

    returncode = run_all.main([
        "--allow_random_init",
        "--output_dir", str(tmp_path / "results"),
        *skip_args,
    ])

    assert returncode == 1
    assert reported["stage_status"][stage] == "failed"


def test_run_all_marks_unreadable_requested_input_as_failed(monkeypatch, tmp_path):
    lm_data = tmp_path / "lm.txt"
    lm_data.write_text("sample", encoding="utf-8")
    reported = {}
    evaluator_calls = []
    real_open = open

    def deny_lm_input(path, *args, **kwargs):
        if Path(path) == lm_data:
            raise PermissionError("input is not readable")
        return real_open(path, *args, **kwargs)

    def fake_run_script(_script_path, args_list):
        evaluator_calls.append(args_list)
        output_path = Path(args_list[args_list.index("--output_path") + 1])
        output_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        run_all,
        "load_config",
        lambda _: {"data": {"lm_eval_path": str(lm_data)}},
    )
    monkeypatch.setattr(run_all, "open", deny_lm_input, raising=False)
    monkeypatch.setattr(run_all, "run_script", fake_run_script)
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )

    returncode = run_all.main([
        "--allow_random_init",
        "--output_dir", str(tmp_path / "results"),
        "--skip_qa", "--skip_generation", "--skip_speed",
    ])

    assert returncode == 1
    assert evaluator_calls == []
    assert reported["stage_status"]["lm"] == "failed"


def test_run_all_uses_yaml_dtype_and_output_dir_when_cli_omits_them(
    monkeypatch, tmp_path
):
    configured_output_dir = tmp_path / "configured-results"
    commands = []
    reported = {}

    def fake_run_script(_script_path, args_list):
        commands.append(args_list)
        output_path = Path(args_list[args_list.index("--output_path") + 1])
        output_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_all,
        "load_config",
        lambda _: {
            "model": {"dtype": "bf16"},
            "eval": {"output_dir": str(configured_output_dir)},
        },
    )
    monkeypatch.setattr(run_all, "run_script", fake_run_script)
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )

    assert run_all.main([
        "--allow_random_init",
        "--skip_lm", "--skip_qa", "--skip_generation",
    ]) == 0

    speed_args = commands[0]
    assert speed_args[speed_args.index("--dtype") + 1] == "bf16"
    assert Path(speed_args[speed_args.index("--output_path") + 1]).parent == configured_output_dir
    assert reported["output_dir"] == str(configured_output_dir)
    assert reported["dtype"] == "bf16"


def test_run_all_explicit_cli_dtype_and_output_dir_override_yaml(monkeypatch, tmp_path):
    configured_output_dir = tmp_path / "configured-results"
    cli_output_dir = tmp_path / "cli-results"
    commands = []
    reported = {}

    def fake_run_script(_script_path, args_list):
        commands.append(args_list)
        output_path = Path(args_list[args_list.index("--output_path") + 1])
        output_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        run_all,
        "load_config",
        lambda _: {
            "model": {"dtype": "bf16"},
            "eval": {"output_dir": str(configured_output_dir)},
        },
    )
    monkeypatch.setattr(run_all, "run_script", fake_run_script)
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )

    assert run_all.main([
        "--allow_random_init",
        "--dtype", "fp32",
        "--output_dir", str(cli_output_dir),
        "--skip_lm", "--skip_qa", "--skip_generation",
    ]) == 0

    speed_args = commands[0]
    assert speed_args[speed_args.index("--dtype") + 1] == "fp32"
    assert Path(speed_args[speed_args.index("--output_path") + 1]).parent == cli_output_dir
    assert reported["output_dir"] == str(cli_output_dir)
    assert reported["dtype"] == "fp32"


@pytest.mark.parametrize(
    ("stage", "output_name", "skip_args"),
    [
        ("lm", "lm_eval.json", ["--skip_qa", "--skip_generation", "--skip_speed"]),
        ("qa", "qa_eval.json", ["--skip_lm", "--skip_generation", "--skip_speed"]),
        (
            "generation",
            "generation_eval.json",
            ["--skip_lm", "--skip_qa", "--skip_speed"],
        ),
        ("speed", "speed_eval.json", ["--skip_lm", "--skip_qa", "--skip_generation"]),
    ],
)
def test_run_all_does_not_accept_stale_output_when_evaluator_writes_nothing(
    monkeypatch, tmp_path, stage, output_name, skip_args
):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    stale_output = output_dir / output_name
    stale_output.write_text("{}", encoding="utf-8")
    data_paths = {}
    for config_key in (
        "lm_eval_path",
        "qa_eval_path",
        "generation_eval_path",
    ):
        data_path = tmp_path / f"{config_key}.jsonl"
        data_path.write_text("{}\n", encoding="utf-8")
        data_paths[config_key] = str(data_path)
    reported = {}

    monkeypatch.setattr(
        run_all,
        "load_config",
        lambda _: {"data": data_paths},
    )
    monkeypatch.setattr(run_all, "run_script", lambda *_: 0)
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )

    returncode = run_all.main([
        "--allow_random_init",
        "--output_dir", str(output_dir),
        *skip_args,
    ])

    assert returncode == 1
    assert reported["stage_status"][stage] == "failed"


@pytest.mark.parametrize(
    "config_text",
    [
        "[]\n",
        "model: []\n",
        "eval:\n  output_dir: []\n",
        "model: [\n",
    ],
)
def test_run_all_reports_invalid_yaml_config_as_cli_error(tmp_path, config_text):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        run_all.main([
            "--allow_random_init",
            "--config_path", str(config_path),
            "--skip_lm", "--skip_qa", "--skip_generation", "--skip_speed",
        ])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("stage", "data_key", "metrics_name", "predictions_name", "skip_args"),
    [
        (
            "qa",
            "qa_eval_path",
            "qa_eval.json",
            "qa_predictions.jsonl",
            ["--skip_lm", "--skip_generation", "--skip_speed"],
        ),
        (
            "generation",
            "generation_eval_path",
            "generation_eval.json",
            "generation_predictions.jsonl",
            ["--skip_lm", "--skip_qa", "--skip_speed"],
        ),
    ],
)
def test_run_all_requires_fresh_prediction_output(
    monkeypatch,
    tmp_path,
    stage,
    data_key,
    metrics_name,
    predictions_name,
    skip_args,
):
    data_path = tmp_path / "input.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    stale_predictions = output_dir / predictions_name
    stale_predictions.write_text('{"stale": true}\n', encoding="utf-8")
    reported = {}

    def write_metrics_only(_script_path, args_list):
        metrics_path = Path(args_list[args_list.index("--output_path") + 1])
        assert metrics_path.name == metrics_name
        metrics_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        run_all,
        "load_config",
        lambda _: {"data": {data_key: str(data_path)}},
    )
    monkeypatch.setattr(run_all, "run_script", write_metrics_only)
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )

    returncode = run_all.main([
        "--allow_random_init",
        "--output_dir", str(output_dir),
        *skip_args,
    ])

    assert returncode == 1
    assert reported["stage_status"][stage] == "failed"
    assert not stale_predictions.exists()


@pytest.mark.parametrize(
    ("stage", "data_key", "metrics_name", "predictions_name", "skip_args"),
    [
        (
            "qa",
            "qa_eval_path",
            "qa_eval.json",
            "qa_predictions.jsonl",
            ["--skip_lm", "--skip_generation", "--skip_speed"],
        ),
        (
            "generation",
            "generation_eval_path",
            "generation_eval.json",
            "generation_predictions.jsonl",
            ["--skip_lm", "--skip_qa", "--skip_speed"],
        ),
    ],
)
def test_run_all_accepts_fresh_metrics_and_predictions(
    monkeypatch,
    tmp_path,
    stage,
    data_key,
    metrics_name,
    predictions_name,
    skip_args,
):
    data_path = tmp_path / "input.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "results"
    reported = {}

    def write_outputs(_script_path, args_list):
        metrics_path = Path(args_list[args_list.index("--output_path") + 1])
        predictions_path = Path(
            args_list[args_list.index("--predictions_path") + 1]
        )
        assert metrics_path.name == metrics_name
        assert predictions_path.name == predictions_name
        metrics_path.write_text("{}", encoding="utf-8")
        predictions_path.write_text("{}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        run_all,
        "load_config",
        lambda _: {"data": {data_key: str(data_path)}},
    )
    monkeypatch.setattr(run_all, "run_script", write_outputs)
    monkeypatch.setattr(
        run_all,
        "generate_markdown_report",
        lambda **kwargs: reported.update(kwargs),
    )

    assert run_all.main([
        "--allow_random_init",
        "--output_dir", str(output_dir),
        *skip_args,
    ]) == 0
    assert reported["stage_status"][stage] == "success"
