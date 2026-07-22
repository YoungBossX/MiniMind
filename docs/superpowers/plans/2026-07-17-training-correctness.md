# Training Correctness Remediation Implementation Plan

> Status (2026-07-22): Historical implementation record. Unchecked boxes preserve the original plan and are not active tasks; the remediation is included in `09196d7`. Use the current code, tests, and root README for runtime behavior.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MiniMind's supported training paths mathematically correct, resume-safe, padding-efficient, and evaluation-safe.

**Architecture:** Add small pure helpers at existing ownership boundaries, then wire them into the current trainers. Preserve public commands while failing closed for unsafe configurations and incompatible state.

**Tech Stack:** Python, PyTorch, pytest, Hugging Face tokenizer/GenerationMixin.

---

### Task 1: RL objective correctness

**Files:**
- Modify: `trainer/train_ppo.py`
- Modify: `trainer/train_grpo.py`
- Test: `tests/test_training_correctness.py`

- [ ] Add failing tests proving padded values cannot change valid GAE, sampled KL uses `log_ref - log_policy`, temperature changes stored log-probs, GRPO clipping can leave ratio 1, and rank synchronization returns one decision.
- [ ] Run `python -m pytest tests/test_training_correctness.py -q` and confirm the new tests fail for the expected assertions.
- [ ] Implement terminal-masked GAE, correctly oriented sampled KL, temperature-scaled log-probs, synchronized PPO early stopping, and repeated GRPO updates against detached rollout log-probs.
- [ ] Validate `temperature > 0`, `num_generations >= 2`, `grpo_epochs >= 1`, and reject unscaled FP16 RL.
- [ ] Re-run the focused tests and the offline PPO/GRPO smoke tests.

### Task 2: Supervision-preserving tokenization

**Files:**
- Modify: `dataset/llm_dataset.py`
- Modify: `trainer/train_full_sft.py`
- Modify: `trainer/train_reason.py`
- Modify: `trainer/train_dpo.py`
- Test: `tests/test_runtime_dynamic_padding.py`

- [ ] Add failing synthetic tests with a prompt longer than the limit and assert that assistant target/EOS tokens survive; add malformed examples that must raise instead of returning zero masks.
- [ ] Add a pretraining test asserting EOS is present after truncation for raw text without a boundary.
- [ ] Implement final-assistant-aware cropping and explicit nonzero-target validation for SFT and both DPO branches.
- [ ] Set SFT default context to 512 and reasoning default context to 1024; align DPO default learning rate with the documented `1e-6` recipe.
- [ ] Run dynamic-padding, data, and training correctness tests.

### Task 3: Resume and full-precision handoff

**Files:**
- Modify: `trainer/trainer_utils.py`
- Modify: trainers that call `lm_checkpoint`
- Test: `tests/test_runtime_training_helpers.py`
- Test: `tests/test_runtime_model_loading.py`

- [ ] Add failing tests for compile-wrapper state keys, incompatible world size/metadata, locked atomic replacement, and full-precision resume-state preference.
- [ ] Add recursive model unwrapping and atomic checkpoint installation with `os.replace`.
- [ ] Store and validate dataset path/stat, batch size, accumulation, sequence length, world size, and compile mode.
- [ ] Remove world-size step conversion and default new experiments to `--from_resume 0`.
- [ ] Prefer the previous stage's full-precision resume model during `init_model`.
- [ ] Run checkpoint/model-loading tests.

### Task 4: Masked SDPA and padding-aware MoE

**Files:**
- Modify: `model/MiniMindModel.py`
- Test: `tests/test_training_correctness.py`

- [ ] Add failing equivalence tests for padded right/left batches under SDPA versus the manual path, plus a test that MoE auxiliary loss ignores appended padding.
- [ ] Route non-cached multi-token attention through SDPA with a boolean key mask and causal mode, avoiding per-layer CUDA boolean synchronization.
- [ ] Pass the valid-token mask into MoE auxiliary-statistics calculations.
- [ ] Run model correctness, cached generation, and GPU smoke tests.

### Task 5: Evaluation and feedback isolation

**Files:**
- Modify: `evals/run_all.py`
- Modify: `feedback/build_feedback_dataset.py`
- Test: `tests/test_runtime_evaluation_orchestration.py`
- Test: `tests/test_feedback_loop.py`

- [ ] Add failing tests requiring explicit random-init smoke mode and preventing references from creating ready SFT rows.
- [ ] Add `--allow_random_init` to aggregate evaluation; otherwise reject an empty checkpoint.
- [ ] Keep evaluation-derived references in metadata while emitting an empty assistant target with `needs_review` status.
- [ ] Run evaluation/feedback tests.

### Task 6: Integrated verification

**Files:**
- Modify: `README.md` only for changed defaults and MoE parameter count.

- [ ] Run `E:\Anaconda\envs\pytorch_env\python.exe -m pytest -q`.
- [ ] Run all offline smoke stages, including PPO and GRPO with the local reward model.
- [ ] Confirm `git diff --check`, inspect the complete diff, and remove generated caches/reports.
- [ ] Request an independent final code review and resolve all important findings.
