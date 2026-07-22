# Training Correctness Repairs Implementation Plan

> Status (2026-07-22): Historical implementation record. Unchecked boxes preserve the original plan and are not active tasks; use the current code, tests, and root README for runtime behavior.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RL training, checkpoint resume, padding, and evaluation outcomes faithful to their documented semantics.

**Architecture:** Preserve the existing scripts and introduce only small shared helpers for accumulation boundaries. PPO owns its per-rollout optimizer updates; other trainers retain batch accumulation but save only after completed optimizer updates. The model accepts mask-derived RoPE positions so left-padded rollouts are equivalent to unpadded inputs.

**Tech Stack:** Python, PyTorch, Transformers, pytest.

---

### Task 1: Lock the repair contract with regression tests

**Files:**
- Create: `tests/test_training_correctness.py`
- Test: `tests/test_training_correctness.py`

- [ ] **Step 1: Write failing source-level tests**

```python
def test_ppo_steps_inside_ppo_epoch_loop():
    text = read("trainer/train_ppo.py")
    loop = text[text.index("for ppo_ep in range(args.ppo_epochs):"):text.index("return metrics", text.index("for ppo_ep"))]
    assert "actor_optimizer.step()" in loop
    assert "critic_optimizer.step()" in loop

def test_model_uses_position_ids_and_grpo_passes_attention_mask():
    assert "position_ids" in read("model/MiniMindModel.py")
    assert "attention_mask=attention_mask" in read("trainer/train_grpo.py")
```

- [ ] **Step 2: Run the test and observe failure**

Run: `python -m pytest tests/test_training_correctness.py -q -p no:cacheprovider`

Expected: failures identifying the absent PPO optimizer calls and GRPO mask propagation.

- [ ] **Step 3: Extend tests for resume/evaluation/schema invariants**

```python
def test_checkpoint_saves_only_after_optimizer_step():
    text = read("trainer/train_full_sft.py")
    assert "did_optimizer_step" in text

def test_evaluator_rejects_missing_checkpoint_and_rlaif_supports_prompt():
    assert "raise FileNotFoundError" in read("evals/core/load_model.py")
    assert "sample.get(\"prompt\")" in read("dataset/llm_dataset.py")
```

### Task 2: Repair PPO update semantics

**Files:**
- Modify: `trainer/train_ppo.py:336-544`
- Test: `tests/test_training_correctness.py`

- [ ] **Step 1: Make the failing PPO contract test pass**

Inside `ppo_update`, place the complete update after `loss.backward()` and before the next PPO epoch:

```python
clip_grad_norm_(actor_model.parameters(), args.grad_clip)
clip_grad_norm_(critic_model.parameters(), args.grad_clip)
actor_optimizer.step()
critic_optimizer.step()
actor_scheduler.step()
critic_scheduler.step()
actor_optimizer.zero_grad(set_to_none=True)
critic_optimizer.zero_grad(set_to_none=True)
```

Remove the outer optimizer step in `ppo_train_epoch`, remove division by `accumulation_steps` from PPO loss, and reject `--accumulation_steps != 1` before building PPO optimizers. Set scheduler `T_max` to `iters * args.ppo_epochs * args.epochs`.

- [ ] **Step 2: Run targeted regression test**

Run: `python -m pytest tests/test_training_correctness.py -q -p no:cacheprovider`

Expected: PPO source contract passes.

### Task 3: Make left padding position-safe and mask GRPO forwards

**Files:**
- Modify: `model/MiniMindModel.py:167-294,704-800`
- Modify: `trainer/train_grpo.py:30-149`
- Test: `tests/test_training_correctness.py`

- [ ] **Step 1: Add explicit RoPE position support**

```python
def _position_ids(attention_mask, seq_length, start_pos, device):
    if attention_mask is None:
        return torch.arange(start_pos, start_pos + seq_length, device=device).unsqueeze(0)
    positions = attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(attention_mask == 0, 0)
    return positions[:, -seq_length:]
```

Pass `position_ids` from `MiniMindForCausalLM.forward` through `MiniMindModel`, `MiniMindBlock`, and `Attention`. In `apply_rotary_pos_emb`, index the frequency buffers by those ids and broadcast as `[batch, sequence, 1, head_dim]`.

- [ ] **Step 2: Pass the generated full attention mask in GRPO**

```python
attention_mask = (outputs != tokenizer.pad_token_id).long()
actor_logps, entropy_per_token = get_per_token_logps(model, outputs, R, attention_mask)
ref_logps, _ = get_per_token_logps(ref_model, outputs, R, attention_mask)
```

Make `get_per_token_logps` accept the mask and pass it to the model.

- [ ] **Step 3: Run targeted regression test**

Run: `python -m pytest tests/test_training_correctness.py -q -p no:cacheprovider`

Expected: position and GRPO-mask assertions pass.

### Task 4: Preserve accumulation semantics at checkpoint boundaries

**Files:**
- Modify: `trainer/trainer_utils.py:21-30`
- Modify: `trainer/train_pretrain.py`, `trainer/train_full_sft.py`, `trainer/train_lora.py`, `trainer/train_dpo.py`, `trainer/train_reason.py`, `trainer/train_grpo.py`
- Test: `tests/test_training_correctness.py`

- [ ] **Step 1: Add shared boundary helpers**

```python
def accumulation_window_size(step, total_steps, accumulation_steps):
    window_start = ((step - 1) // accumulation_steps) * accumulation_steps + 1
    return min(accumulation_steps, total_steps - window_start + 1)

def should_optimizer_step(step, total_steps, accumulation_steps):
    return step % accumulation_steps == 0 or step == total_steps
```

- [ ] **Step 2: Use the helpers in each non-PPO trainer**

Replace `loss / args.accumulation_steps` with:

```python
loss = loss / accumulation_window_size(step, iters, args.accumulation_steps)
did_optimizer_step = should_optimizer_step(step, iters, args.accumulation_steps)
```

Use `did_optimizer_step` for optimizer/scheduler steps. Gate each checkpoint by `did_optimizer_step` so a saved resume state never excludes live gradients.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -q -p no:cacheprovider`

Expected: all tests pass.

### Task 5: Make data, loading, and evaluation fail safely

**Files:**
- Modify: `dataset/llm_dataset.py:401-405`
- Modify: `trainer/trainer_utils.py:204-214`
- Modify: `evals/core/load_model.py:57-60`
- Modify: `evals/run_all.py:29-39,102-166`
- Modify: `evals/eval_speed.py:620-675`
- Test: `tests/test_training_correctness.py`

- [ ] **Step 1: Support audited raw RLAIF prompts**

```python
if isinstance(sample.get("prompt"), str) and sample["prompt"].strip():
    messages = [{"role": "user", "content": sample["prompt"]}]
    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "answer": sample.get("answer", "")}
```

- [ ] **Step 2: Reject unsafe loads and stale evaluator output**

Use strict model loading in `init_model`. In the evaluator loader, raise `FileNotFoundError` when a non-empty checkpoint path does not exist. In `run_all`, retain an output only when `run_script(...) == 0`.

- [ ] **Step 3: Measure GPU time and batches correctly**

```python
if device.startswith("cuda"):
    torch.cuda.synchronize(device)
t0 = time.perf_counter()
gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
if device.startswith("cuda"):
    torch.cuda.synchronize(device)
```

Build `inputs` from a list containing `batch_size` prompts and count generated tokens across that batch.

- [ ] **Step 4: Run verification**

Run: `python -m pytest -q -p no:cacheprovider && python -m compileall -q model trainer dataset eval evals feedback scripts tests`

Expected: pytest exits zero and compilation emits no errors.
