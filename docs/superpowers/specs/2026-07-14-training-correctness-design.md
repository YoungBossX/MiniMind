# Training correctness repairs

> Status (2026-07-22): Historical design record. Decisions may have been superseded by later hardening; use the current code, tests, and root README for runtime behavior.

## Scope

Repair the correctness risks identified in PPO/GRPO, supervised training resume,
padding handling, data validation, checkpoint loading, and evaluation utilities.
Do not redesign model architecture or add a new training framework.

## Decisions

1. PPO performs one optimizer and scheduler update per PPO epoch for a rollout.
   `accumulation_steps` is rejected unless it is one, because accumulating across
   rollouts prevents PPO's policy-ratio epochs from observing updated parameters.
2. Supervised, DPO, GRPO, and reasoning trainers use the true size of their final
   accumulation window. Checkpoints are emitted only after an optimizer update,
   so they never claim resumability while gradients are only partially accumulated.
3. The model derives RoPE positions from `attention_mask` and accepts explicit
   `position_ids`. GRPO passes masks whenever it recomputes sequence log-probs.
4. RLAIF accepts either a conversation list or a raw prompt. Raw prompts are
   rendered with the tokenizer's ChatML template before rollout.
5. A supplied checkpoint path must exist. The evaluation runner consumes an
   evaluator's output only when that evaluator exits successfully. GPU timing is
   synchronized, and the configured speed batch size is used.
6. Base-weight loading is strict to prevent silent partial initialization.

## Verification

Add focused regression tests for source-level invariants that can run without
PyTorch, then run the existing pytest suite and Python compilation. Model-level
numerical smoke tests remain conditional on a Python environment with PyTorch.
