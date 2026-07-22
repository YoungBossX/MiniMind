from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.sh"


def test_run_eval_delegates_once_to_run_all_without_duplicate_stages():
    script = SCRIPT.read_text(encoding="utf-8")

    assert script.count("python evals/run_all.py") == 1
    assert "python evals/eval_lm.py" not in script
    assert "python evals/eval_qa.py" not in script
    assert "python evals/eval_generation.py" not in script
    assert "python evals/eval_speed.py" not in script
    assert "|| true" not in script
