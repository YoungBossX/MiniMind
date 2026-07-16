import sys
from pathlib import Path

from evals import run_all
from evals.core.metrics import compute_speed_metrics


def test_run_all_returns_nonzero_when_a_requested_evaluator_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all, "run_script", lambda *_: 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all.py",
            "--output_dir",
            str(tmp_path),
            "--skip_qa",
            "--skip_generation",
            "--skip_speed",
        ],
    )

    assert run_all.main() == 1


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
