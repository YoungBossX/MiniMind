# Adversarial Remediation Implementation Plan

> Status (2026-07-22): Historical implementation record. Unchecked boxes preserve the original plan and are not active tasks; use the current code, tests, and root README for runtime behavior.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Make mid-epoch resume, evaluation orchestration, PPO critic inputs, adapter loading, and DPO data auditing fail-safe and testable.

**Architecture:** A shared epoch-batch helper will make normal and resumed training consume one deterministic batch stream. Evaluation orchestration will carry explicit stage state to the report and process exit status. Local validation functions will reject ambiguous adapters and malformed preference pairs before they influence model quality.

**Tech Stack:** Python 3.10, PyTorch 2.5, pytest, AST regression tests, Hugging Face Transformers.

---

### Task 1: Deterministic pretrain, LoRA, and DPO batch streams

**Files:**
- Modify: trainer/trainer_utils.py
- Modify: trainer/train_pretrain.py:364-395
- Modify: trainer/train_lora.py:377-397
- Modify: trainer/train_dpo.py:470-490
- Modify: tests/test_training_correctness.py

- [ ] **Step 1: Write the failing sampler regression test**

~~~python
def test_epoch_batch_sampler_resume_is_a_suffix_of_the_normal_stream():
    source = list(range(8))
    normal = list(iter_epoch_batches(source, batch_size=2, epoch=7, skip_batches=0))
    resumed = list(iter_epoch_batches(source, batch_size=2, epoch=7, skip_batches=1))
    assert resumed == normal[1:]
~~~

The test must import the real helper when PyTorch is installed and be skipped otherwise.

- [ ] **Step 2: Run test to verify it fails**

Run:

~~~powershell
E:\Anaconda\envs\pytorch_env\python.exe -B -m pytest tests/test_training_correctness.py -q -p no:cacheprovider
~~~

Expected: failure because the helper does not exist and current resume uses sequential indices.

- [ ] **Step 3: Implement the shared sampler helper**

~~~python
def build_epoch_batch_sampler(dataset_size, batch_size, epoch, skip_batches=0,
                              sampler=None, seed=42):
    if sampler is None:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        sampler = torch.randperm(dataset_size, generator=generator).tolist()
    return SkipBatchSampler(sampler, batch_size, skip_batches)
~~~

Replace both normal and resume DataLoader construction in the three trainers with:

~~~python
batch_sampler = build_epoch_batch_sampler(
    len(train_ds), args.batch_size, epoch, skip_batches=skip,
    sampler=train_sampler,
)
loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                    num_workers=args.num_workers, pin_memory=True)
~~~

- [ ] **Step 4: Run focused and full regression tests**

~~~powershell
E:\Anaconda\envs\pytorch_env\python.exe -B -m pytest -q -p no:cacheprovider
~~~

Expected: focused suffix test and full suite pass.

### Task 2: Propagate evaluation failures and batch configuration

**Files:**
- Modify: evals/run_all.py
- Modify: evals/core/report.py
- Modify: tests/test_training_correctness.py

- [ ] **Step 1: Write failing orchestration tests**

~~~python
def test_run_all_returns_nonzero_when_a_requested_stage_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all, "load_config", lambda _: {"data": {}})
    monkeypatch.setattr(run_all, "run_script", lambda *_: 1)
    assert run_all.main(["--output_dir", str(tmp_path), "--skip_qa",
                         "--skip_generation", "--skip_speed"]) == 1
~~~

Also assert the speed command includes --batch_size with the config value.

- [ ] **Step 2: Run tests to verify failure**

Expected: main returns None and speed arguments omit --batch_size.

- [ ] **Step 3: Implement explicit stage statuses**

~~~python
stage_status = {"lm": "skipped", "qa": "skipped",
                "generation": "skipped", "speed": "skipped"}
~~~

Set each executed stage to success only when return code is zero and its expected output exists; otherwise set failed. Pass stage_status to generate_markdown_report, append a status table, pass:

~~~python
"--batch_size", str(eval_cfg.get("batch_size", 1))
~~~

to speed evaluation, and finish with:

~~~python
return 1 if "failed" in stage_status.values() else 0
~~~

with raise SystemExit(main()) in the module entry point.

- [ ] **Step 4: Verify focused and full tests**

Run the focused orchestration tests and complete pytest suite. Expected: all pass.

### Task 3: Correct PPO critic and explicit LoRA loading

**Files:**
- Modify: trainer/train_ppo.py:57-65
- Modify: evals/core/load_model.py:68-71
- Modify: tests/test_training_correctness.py

- [ ] **Step 1: Write failing tests**

~~~python
with pytest.raises(FileNotFoundError, match="LoRA"):
    load_model_and_tokenizer(lora_path="missing-adapter.pth")
~~~

Also add a source/runtime test showing the critic value head receives the backbone output directly.

- [ ] **Step 2: Run tests to verify failure**

Expected: the source contains self.model.norm(outputs[0]) and a requested absent LoRA path returns a base model.

- [ ] **Step 3: Implement minimal fixes**

~~~python
hidden_states = outputs[0]
~~~

In the eval loader:

~~~python
if lora_path and not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
~~~

- [ ] **Step 4: Verify tests and PPO smoke**

~~~powershell
E:\Anaconda\envs\pytorch_env\python.exe eval/smoke_test.py --stage ppo --device cuda:0
~~~

Expected: critic test passes and PPO smoke reports 6/6 assertions.

### Task 4: Validate DPO preference pairs before training

**Files:**
- Modify: scripts/audit_data.py
- Modify: tests/test_feedback_loop.py

- [ ] **Step 1: Write failing DPO audit tests**

~~~python
sample = {
    "chosen": [{"role": "user", "content": "A"},
               {"role": "assistant", "content": "good"}],
    "rejected": [{"role": "user", "content": "B"},
                 {"role": "assistant", "content": "bad"}],
}
codes = {issue["code"] for issue in validate_sample(sample, "dpo", 1)}
assert "dpo_prompt_mismatch" in codes
~~~

- [ ] **Step 2: Run tests to verify failure**

Expected: current validation returns no issue for this malformed pair.

- [ ] **Step 3: Implement pair validation**

Add validate_conversation and conversation_prefix helpers. Require every turn to have role in {system,user,assistant} and nonblank string content; require both lists to end in assistant; compare chosen[:-1] and rejected[:-1] for exact equality. Emit dpo_invalid_turn, dpo_missing_final_assistant, and dpo_prompt_mismatch issue codes.

- [ ] **Step 4: Verify audit tests**

~~~powershell
E:\Anaconda\envs\pytorch_env\python.exe -B -m pytest -q -p no:cacheprovider
~~~

Expected: all pass.

### Task 5: Record speed batch provenance and complete verification

**Files:**
- Modify: evals/eval_speed.py
- Modify: evals/core/metrics.py
- Modify: tests/test_training_correctness.py

- [ ] **Step 1: Write failing metric test**

~~~python
metrics = compute_speed_metrics([10.0], 8, 4, batch_size=4)
assert metrics["batch_size"] == 4
~~~

- [ ] **Step 2: Run test to verify failure**

Expected: compute_speed_metrics rejects the new keyword and speed JSON has no batch-size provenance.

- [ ] **Step 3: Implement minimal observability**

Add a batch_size parameter to compute_speed_metrics, pass it from evaluate_speed, and include it in JSON. Do not change attention math in this task; use this provenance to make dynamic-padding comparisons meaningful.

- [ ] **Step 4: Run final verification**

~~~powershell
E:\Anaconda\envs\pytorch_env\python.exe -B -m pytest -q -p no:cacheprovider
E:\Anaconda\envs\pytorch_env\python.exe -B -m compileall -q model trainer dataset eval evals feedback scripts tests
git diff --check
~~~

Then run CPU-safe smoke stages and GPU PPO/GRPO stages. Expected: zero test failures, no syntax errors, no whitespace errors, and smoke assertions pass.
