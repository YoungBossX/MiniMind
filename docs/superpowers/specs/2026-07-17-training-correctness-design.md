# Training Correctness Remediation Design

> Status (2026-07-22): Historical design record. The final implementation is in `09196d7`; use the current code, tests, and root README for runtime behavior.

## Scope

This change hardens the existing MiniMind training pipeline without replacing its
PyTorch model or adding a new attention dependency. It addresses confirmed
correctness defects in PPO/GRPO, supervised truncation, checkpoint resume, padded
attention, and evaluation feedback handling.

Training datasets and existing model weights are not rewritten. Data cleanup is
implemented as validation/tooling so that destructive dataset rewrites remain an
explicit operator action.

## Design

### RL objectives

- GAE uses an explicit response-transition mask. Padding resets the recurrence,
  and terminal response tokens do not bootstrap from padded critic values.
- Sampled k3 KL uses `log_ref - log_policy`, matching samples drawn from the
  policy distribution.
- Generation temperature is a named argument and is also applied when computing
  policy/reference log probabilities and entropy.
- PPO early-stop decisions are synchronized across ranks before any rank can
  skip a DDP backward pass.
- GRPO performs real repeated policy updates against frozen rollout log-probs;
  clipping is therefore meaningful. Unsafe one-member groups and FP16 RL are
  rejected at argument validation.

### Supervised data

- Pretraining explicitly guarantees a terminal EOS token even when a new data
  source does not provide one.
- SFT/DPO tokenization locates the final assistant turn before truncation. It
  preserves the assistant marker, the complete target when it fits, terminal
  EOS, and as much recent prompt context as the remaining budget permits.
- Any sample or DPO branch with zero supervised tokens raises a descriptive
  error instead of producing a zero-loss optimizer step.
- SFT and reasoning defaults are aligned with the 512- and 1024-token dataset
  contracts.

### Checkpoints

- Model wrappers (`DistributedDataParallel` and `torch.compile`) are recursively
  unwrapped before serialization and state loading.
- Checkpoints are written to a temporary sibling and installed with
  `os.replace`. A locked destination raises while preserving the last good file;
  it never falls back to copying over the live checkpoint.
- Resume metadata records training/data identity. World-size or metadata changes
  fail closed instead of rescaling a batch index.
- Cross-stage initialization prefers the full-precision model state in the
  previous stage's resume checkpoint, with the FP16 inference file as fallback.

### Padding and MoE

- PyTorch SDPA combines causal and padding masks and passes `is_causal=False` for
  PyTorch 2.5 compatibility. Cached single-token generation keeps the existing
  manual path. This final behavior supersedes the earlier simplified
  `is_causal=True` design.
- MoE auxiliary load-balancing statistics exclude padded tokens while routed
  outputs remain shape-compatible.

### Evaluation safety

- Aggregate evaluation requires a real checkpoint unless an explicit smoke-mode
  flag permits random initialization.
- Feedback candidates derived from evaluation references remain review-only and
  never become ready-to-train SFT rows automatically.

## Compatibility

Existing command names and model checkpoint formats remain supported. Old resume
files without metadata can be used only through an explicit legacy override;
new experiments default to no automatic resume. Existing inference `.pth` files
remain loadable.

## Verification

Every confirmed defect receives a focused regression test. The full offline test
suite and targeted PPO/GRPO smoke tests run in `pytorch_env`. GPU checks compare
masked SDPA with the manual implementation on padded batches and exercise cached
generation.
