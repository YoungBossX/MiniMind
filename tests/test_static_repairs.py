import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def trainer_scripts():
    return sorted(TRAINER.glob("train_*.py"))


def test_training_script_defaults_are_project_root_relative():
    bad_defaults = []
    for path in trainer_scripts():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r'default=([\'"])\.\./', text):
            line = text[: match.start()].count("\n") + 1
            bad_defaults.append(f"{path.name}:{line}")

    assert bad_defaults == []


def test_checkpoint_helpers_are_not_hardcoded_to_parent_directory():
    offenders = []
    for path in trainer_scripts():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r'save_dir=([\'"])\.\./checkpoints\1', text):
            line = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}")

    assert offenders == []


def test_learning_rate_schedule_has_warmup_then_cosine():
    text = read("trainer/trainer_utils.py")

    assert "warmup_steps" in text
    assert "current_step < warmup_steps" in text
    assert "progress = (current_step - warmup_steps)" in text


def test_gradient_accumulation_updates_on_epoch_tail():
    offenders = []
    for path in trainer_scripts():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped == "if step % args.accumulation_steps == 0:":
                offenders.append(f"{path.name}:{line_no}")

    assert offenders == []


def test_rl_trainers_save_on_final_step_not_penultimate_step():
    for path in (TRAINER / "train_ppo.py", TRAINER / "train_grpo.py"):
        text = path.read_text(encoding="utf-8")

        assert "step == iters - 1" not in text
        assert "step == iters" in text


def test_grpo_uses_sampled_policy_logps_for_ratio():
    text = read("trainer/train_grpo.py")

    assert "actor_logps - actor_logps.detach()" not in text
    assert "old_logps" in text
    assert "actor_logps - old_logps" in text


def test_ppo_old_logp_comes_from_rollout_actor():
    text = read("trainer/train_ppo.py")

    assert "old_logits = old_actor_model" not in text
    assert "sampled_policy_logits" in text
    assert "old_logp = F.log_softmax(sampled_policy_logits" in text


def test_rl_smoke_generation_does_not_pass_token_type_ids():
    text = read("eval/smoke_test.py")

    assert text.count("return_token_type_ids=False") >= 2
    assert "old_actor, _ = init_model" not in text


def test_dpo_exposes_length_normalized_logprob_reduction():
    text = read("trainer/train_dpo.py")

    assert "logprob_reduction" in text
    assert "choices=[\"mean\", \"sum\"]" in text
    assert "seq_lengths = mask.sum" in text


def test_dpo_smoke_asserts_relative_margin_not_raw_logp_order():
    text = read("eval/smoke_test.py")

    assert 'assertion("chosen_gt_rejected"' not in text
    assert 'assertion("dpo_margin_gt_0"' in text


def test_benchmark_assertions_are_not_tautologies():
    text = read("eval/benchmark.py")

    assert 'gen_metrics["eos_rate"] >= 0' not in text
    assert 'format_metrics["tag_complete"] >= 0' not in text


def test_eval_native_loader_requires_explicit_model_path():
    text = read("eval.py")

    assert 'if "model" in args.load_from:' not in text
    assert "native_model_dir" in text
    assert "is_native_model" in text


def test_project_declares_python_dependencies():
    assert (ROOT / "requirements.txt").exists()


def test_pytest_collects_regression_tests_not_cli_smoke_scripts():
    text = read("pytest.ini")

    assert "testpaths = tests" in text
    assert "python_files = test_*.py" in text
