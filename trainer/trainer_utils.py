import os
import sys
import random
import math
import hashlib
import importlib
import copy
import re
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from trainer.path_utils import CHECKPOINT_DIR, project_path


_ARCHITECTURE_METADATA_FIELDS = (
    "model_type",
    "dropout",
    "bos_token_id",
    "eos_token_id",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "inference_rope_scaling",
    "flash_attention",
    "use_moe",
    "num_experts_per_tok",
    "n_routed_experts",
    "n_shared_experts",
    "scoring_func",
    "aux_loss_alpha",
    "seq_aux",
    "norm_topk_prob",
)

_TRAINING_METADATA_FIELDS = (
    "epochs",
    "learning_rate",
    "critic_learning_rate",
    "grad_clip",
    "num_workers",
    "from_weight",
    "temperature",
    "reasoning",
    "reward_model_path",
    "num_generations",
    "grpo_epochs",
    "ppo_epochs",
    "gamma",
    "gae_lambda",
    "beta",
    "clip_epsilon",
    "entropy_coef",
    "kl_coef",
    "vf_coef",
    "target_kl",
    "loss_type",
    "tag_penalty_weight",
    "logprob_reduction",
)

_MAX_ABS_LOG_RATIO = 20.0
_CHATML_GENERATION_MARKER = "<|im_start|>assistant\n"
_CHATML_MESSAGE_PATTERN = re.compile(
    r"<\|im_start\|>(?P<role>[^\n]+)\n"
    r"(?P<content>.*?)<\|im_end\|>\n?",
    re.DOTALL,
)

_MODEL_FINGERPRINT_EXCLUDED_DIRS = {
    "__pycache__",
    "eval_results",
    "reports",
    "reward_bench_results",
}

_REFERENCE_SIDECAR_CACHE = {}
_HF_MODULE_CACHE_LOCK = threading.RLock()
_CURRENT_CHECKPOINT_SCHEMA = 2
_SUPPORTED_LEGACY_METADATA_SCHEMAS = {1}
_KNOWN_ARTIFACT_STAGES = {
    "pretrain": "pretrain",
    "full_sft": "full_sft",
    "dpo": "dpo",
    "ppo_actor": "ppo",
    "ppo": "ppo",
    "grpo": "grpo",
    "reason": "reason",
}
_RESUME_IDENTITY_FIELDS = (
    "artifact_name",
    "stage",
    "dataset",
    "batch_size",
    "accumulation_steps",
    "max_seq_len",
    "max_gen_len",
    "dtype",
    "compile",
    "world_size",
    "runtime",
    "training",
    "architecture",
)


def _normalized_project_path(path):
    return os.path.normcase(
        os.path.realpath(os.path.abspath(project_path(path)))
    )


def fingerprint_path(path):
    """Return a deterministic content fingerprint for a file or directory."""
    normalized_path = _normalized_project_path(path)
    if not os.path.exists(normalized_path):
        return {
            "kind": "unresolved",
            "identifier": str(path),
        }

    if os.path.isfile(normalized_path):
        root = os.path.dirname(normalized_path)
        files = [("", normalized_path)]
        kind = "file"
    else:
        root = normalized_path
        files = []

        def collect_files(current_path, relative_prefix, ancestors):
            real_path = os.path.normcase(os.path.realpath(current_path))
            if real_path in ancestors:
                raise ValueError(
                    f"Directory link cycle while fingerprinting {current_path!r}"
                )
            with os.scandir(current_path) as iterator:
                entries = sorted(list(iterator), key=lambda entry: entry.name)
            next_ancestors = ancestors | {real_path}
            for entry in entries:
                lower_name = entry.name.lower()
                relative_path = (
                    f"{relative_prefix}/{entry.name}"
                    if relative_prefix
                    else entry.name
                )
                if entry.is_dir(follow_symlinks=True):
                    if (
                        entry.name.startswith(".")
                        or lower_name in _MODEL_FINGERPRINT_EXCLUDED_DIRS
                    ):
                        continue
                    collect_files(entry.path, relative_path, next_ancestors)
                    continue
                if entry.is_file(follow_symlinks=True):
                    if (
                        entry.name.startswith(".")
                        or lower_name.startswith("readme")
                        or lower_name.startswith("license")
                        or lower_name.endswith((".pyc", ".pyo"))
                    ):
                        continue
                    files.append((relative_path.replace(os.sep, "/"), entry.path))
                    continue
                raise ValueError(
                    f"Unsupported or broken filesystem entry: {entry.path!r}"
                )

        collect_files(root, "", set())
        kind = "directory"

    digest = hashlib.sha256()
    total_size = 0
    for relative_path, file_path in files:
        before_stat = os.stat(file_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with open(file_path, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                digest.update(chunk)
        digest.update(b"\0")
        after_stat = os.stat(file_path)
        if (
            before_stat.st_size != after_stat.st_size
            or before_stat.st_mtime_ns != after_stat.st_mtime_ns
            or before_stat.st_ino != after_stat.st_ino
        ):
            raise RuntimeError(
                f"Training dependency changed while hashing: {file_path!r}"
            )

    return {
        "kind": kind,
        "file_count": len(files),
        "size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def distributed_fingerprint_path(path):
    """Hash once per node and reject divergent node-local snapshots."""
    if not dist.is_initialized():
        return fingerprint_path(path)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostnames = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())
    host_ranks = [
        candidate_rank
        for candidate_rank, hostname in enumerate(hostnames)
        if hostname == hostnames[rank]
    ]
    should_hash = rank == min(host_ranks)
    local_payload = None
    if should_hash:
        try:
            local_payload = {"fingerprint": fingerprint_path(path)}
        except Exception as error:
            local_payload = {
                "error": f"{type(error).__name__}: {error}",
            }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_payload)
    node_payloads = [payload for payload in gathered if payload is not None]
    if not node_payloads:
        raise RuntimeError("No node produced a training dependency fingerprint")
    errors = [payload["error"] for payload in node_payloads if "error" in payload]
    if errors:
        raise RuntimeError(
            f"Could not fingerprint shared training dependency: "
            f"{errors[0]}"
        )
    fingerprints = [payload["fingerprint"] for payload in node_payloads]
    if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]):
        raise ValueError(
            "Training dependency snapshots have different content across nodes."
        )
    return fingerprints[0]


class _ClampLogRatioStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_ratio):
        return torch.clamp(
            log_ratio, min=-_MAX_ABS_LOG_RATIO, max=_MAX_ABS_LOG_RATIO
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def clamp_log_ratio(log_ratio):
    """Bound exponent inputs while preserving gradients through saturation."""
    return _ClampLogRatioStraightThrough.apply(log_ratio)


def _token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = (
        encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    )
    if torch.is_tensor(input_ids):
        input_ids = input_ids.tolist()
    return input_ids


def _parse_chatml_prompt(prompt):
    marker_start = prompt.rfind(_CHATML_GENERATION_MARKER)
    if (
        marker_start < 0
        or prompt[marker_start:] != _CHATML_GENERATION_MARKER
    ):
        return None

    body = prompt[:marker_start]
    messages = []
    position = 0
    for match in _CHATML_MESSAGE_PATTERN.finditer(body):
        if match.start() != position:
            return None
        messages.append(
            {
                "segment": match.group(0),
                "role": match.group("role"),
                "content": match.group("content"),
            }
        )
        position = match.end()
    if position != len(body) or not messages:
        return None
    return messages


def _split_chatml_prompt(prompt):
    messages = _parse_chatml_prompt(prompt)
    if messages is None:
        return None
    return [message["segment"] for message in messages]


def _is_lossless_byte_bpe_tokenizer(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if (
        backend is None
        or getattr(backend, "normalizer", None) is not None
        or type(getattr(backend, "model", None)).__name__ != "BPE"
        or type(getattr(backend, "pre_tokenizer", None)).__name__ != "ByteLevel"
        or type(getattr(backend, "decoder", None)).__name__ != "ByteLevel"
        or not hasattr(tokenizer, "get_vocab")
    ):
        return False
    alphabet = getattr(backend.pre_tokenizer, "alphabet", None)
    if not callable(alphabet):
        return False
    byte_alphabet = set(alphabet())
    return len(byte_alphabet) == 256 and byte_alphabet.issubset(tokenizer.get_vocab())


def chatml_prompt_messages(prompt):
    """Extract exact role/content payloads from an actor generation prompt."""
    messages = _parse_chatml_prompt(prompt)
    if messages is None:
        raise ValueError("RL prompt is not a complete ChatML generation prompt")
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]


def _truncate_rl_prompt(tokenizer, prompt, max_length):
    if not max_length or len(_token_ids(tokenizer, prompt)) <= max_length:
        return prompt

    messages = _split_chatml_prompt(prompt)
    if messages is None:
        raise ValueError("RL prompt is not a complete ChatML generation prompt")

    retained = []
    for message in reversed(messages):
        candidate = message + "".join(retained) + _CHATML_GENERATION_MARKER
        if len(_token_ids(tokenizer, candidate)) > max_length:
            break
        retained.insert(0, message)

    if retained:
        return "".join(retained) + _CHATML_GENERATION_MARKER

    newest_message = messages[-1]
    header_end = newest_message.find("\n") + 1
    end_marker = (
        "<|im_end|>\n"
        if newest_message.endswith("<|im_end|>\n")
        else "<|im_end|>"
    )
    header = newest_message[:header_end]
    content = newest_message[header_end:-len(end_marker)]

    def candidate(content_start):
        return (
            header
            + content[content_start:]
            + end_marker
            + _CHATML_GENERATION_MARKER
        )

    if len(_token_ids(tokenizer, candidate(len(content)))) > max_length:
        raise ValueError("RL prompt token budget cannot fit ChatML boundaries and content")

    search_start = 0
    if _is_lossless_byte_bpe_tokenizer(tokenizer):
        vocab = tokenizer.get_vocab()
        max_token_bytes = max(
            (len(token.encode("utf-8")) for token in vocab), default=1
        )
        suffix_byte_budget = max_length * max_token_bytes
        suffix_bytes = 0
        search_start = len(content)
        while search_start > 0:
            char_bytes = len(content[search_start - 1].encode("utf-8"))
            if suffix_bytes + char_bytes > suffix_byte_budget:
                break
            suffix_bytes += char_bytes
            search_start -= 1

    candidate_batch_size = 256
    for batch_start in range(search_start, len(content) + 1, candidate_batch_size):
        boundaries = list(
            range(
                batch_start,
                min(batch_start + candidate_batch_size, len(content) + 1),
            )
        )
        candidate_texts = [candidate(boundary) for boundary in boundaries]
        encoded = tokenizer(
            candidate_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        token_lengths = (
            encoded["length"] if isinstance(encoded, dict) else encoded.length
        )
        if torch.is_tensor(token_lengths):
            token_lengths = token_lengths.tolist()
        for boundary, token_length in zip(boundaries, token_lengths):
            if token_length <= max_length:
                return candidate(boundary)

    raise RuntimeError("RL prompt truncation could not retain a fitting suffix")


def tokenize_rl_prompts(tokenizer, prompts, max_length, device=None):
    """Tokenize complete ChatML rollout prompts without splitting messages."""
    actor_prompts = [
        _truncate_rl_prompt(tokenizer, prompt, max_length) for prompt in prompts
    ]
    previous_padding_side = getattr(tokenizer, "padding_side", None)
    if previous_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            actor_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_token_type_ids=False,
        )
    finally:
        if previous_padding_side is not None:
            tokenizer.padding_side = previous_padding_side
    if device is not None:
        encoded = encoded.to(device)
    return encoded, actor_prompts


def build_rollout_masks(prompt_attention_mask, generated_ids, eos_token_id):
    """Return full attention and generated-action masks through first EOS."""
    if prompt_attention_mask.ndim != 2 or generated_ids.ndim != 2:
        raise ValueError("RL prompt attention and generated IDs must be rank-2")
    prompt_batch, prompt_width = prompt_attention_mask.shape
    generated_batch, generated_width = generated_ids.shape
    if prompt_batch == 0 or generated_batch % prompt_batch != 0:
        raise ValueError("Generated batch must evenly expand the prompt batch")
    if generated_width < prompt_width:
        raise ValueError("Generated sequences are shorter than the prompt width")

    repeats = generated_batch // prompt_batch
    expanded_prompt_mask = prompt_attention_mask.to(generated_ids.device)
    expanded_prompt_mask = expanded_prompt_mask.repeat_interleave(repeats, dim=0)
    completion_ids = generated_ids[:, prompt_width:]
    if eos_token_id is None:
        action_mask = torch.ones_like(completion_ids, dtype=torch.bool)
    else:
        is_eos = completion_ids.eq(eos_token_id)
        is_eos_int = is_eos.to(torch.int64)
        eos_before = is_eos_int.cumsum(dim=1) - is_eos_int
        action_mask = eos_before.eq(0)
    full_attention_mask = torch.cat(
        (
            expanded_prompt_mask,
            action_mask.to(dtype=expanded_prompt_mask.dtype),
        ),
        dim=1,
    )
    return full_attention_mask, action_mask


def clip_gradients(parameters, max_norm):
    """Clip finite gradients, or only validate them when clipping is disabled."""
    max_norm = float(max_norm)
    if math.isnan(max_norm):
        raise ValueError("Gradient max_norm must not be NaN")
    effective_max_norm = math.inf if max_norm <= 0 else max_norm
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        effective_max_norm,
        error_if_nonfinite=True,
    )

# 检查是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def resolve_checkpoint_dir(save_dir, checkpoint_dir=None):
    """Resolve an explicit checkpoint directory or scope it to an output run."""
    if checkpoint_dir:
        return os.path.normpath(project_path(checkpoint_dir))
    if not save_dir:
        raise ValueError("save_dir is required when checkpoint_dir is not explicit")
    return os.path.normpath(os.path.join(project_path(save_dir), "checkpoints"))


def resolve_lora_base_dirs(
    save_dir, base_save_dir=None, base_checkpoint_dir=None
):
    """Resolve LoRA's base-model output independently from adapter checkpoints."""
    adapter_dir = os.path.normpath(project_path(save_dir))
    if base_save_dir:
        resolved_base_dir = os.path.normpath(project_path(base_save_dir))
    elif os.path.normcase(os.path.basename(adapter_dir)) == os.path.normcase("lora"):
        resolved_base_dir = os.path.dirname(adapter_dir)
    else:
        resolved_base_dir = adapter_dir
    return resolved_base_dir, resolve_checkpoint_dir(
        resolved_base_dir, base_checkpoint_dir
    )


def coordinated_checkpoint_save(primary_save, derived_save=None):
    """Run ordered rank-zero saves and propagate failures to every rank."""
    if not dist.is_initialized():
        primary_save()
        if derived_save is not None:
            derived_save()
        return

    rank = dist.get_rank()
    status = [None]
    rank_zero_error = None
    if rank == 0:
        try:
            primary_save()
            if derived_save is not None:
                derived_save()
            status[0] = {"ok": True}
        except BaseException as error:
            rank_zero_error = error
            status[0] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.broadcast_object_list(status, src=0)
    if not status[0]["ok"]:
        if rank == 0 and not isinstance(rank_zero_error, Exception):
            raise rank_zero_error
        raise RuntimeError(
            f"Checkpoint save failed on rank 0: {status[0]['error']}"
        )

# 日志
def Logger(content):
    if is_main_process():
        print(content)

# 动态学习率计算
def get_lr(current_step, total_steps, lr):
    total_steps = max(int(total_steps), 1)
    current_step = min(max(int(current_step), 0), total_steps)
    warmup_steps = max(int(total_steps * 0.1), 1)

    if current_step < warmup_steps:
        return lr * current_step / warmup_steps

    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def accumulation_window_size(step, total_steps, accumulation_steps):
    """Return the number of micro-batches in ``step``'s accumulation window."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    if not 1 <= step <= total_steps:
        raise ValueError("step must be within the current epoch")
    window_start = ((step - 1) // accumulation_steps) * accumulation_steps + 1
    return min(accumulation_steps, total_steps - window_start + 1)


def should_optimizer_step(step, total_steps, accumulation_steps):
    """Whether the current micro-batch closes an accumulation window."""
    return step % accumulation_steps == 0 or step == total_steps


def checkpoint_due(step, total_steps, accumulation_steps, save_interval):
    """Save at the first completed accumulation window after each interval."""
    if save_interval < 1:
        raise ValueError("save_interval must be at least 1")
    window_size = accumulation_window_size(step, total_steps, accumulation_steps)
    previous_step = step - window_size
    return step == total_steps or step // save_interval > previous_step // save_interval

# 初始化分布式
def init_distributed_mode():
    # 非DDP模式
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def temporary_hf_modules_cache(cache_root=None):
    """Redirect trust-remote-code modules to a disposable workspace cache."""
    from transformers import dynamic_module_utils

    root = (
        project_path(".tmp/huggingface_modules")
        if cache_root is None
        else os.fspath(cache_root)
    )
    with _HF_MODULE_CACHE_LOCK:
        os.makedirs(root, exist_ok=True)
        cache_dir = tempfile.mkdtemp(prefix="modules_", dir=root)
        previous_cache = dynamic_module_utils.HF_MODULES_CACHE
        existing_path_entries = sys.path.count(cache_dir)
        body_error = None
        dynamic_module_utils.HF_MODULES_CACHE = cache_dir
        try:
            yield cache_dir
        except BaseException as error:
            body_error = error
            raise
        finally:
            cleanup_errors = []
            try:
                dynamic_module_utils.HF_MODULES_CACHE = previous_cache
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                while sys.path.count(cache_dir) > existing_path_entries:
                    sys.path.remove(cache_dir)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                shutil.rmtree(cache_dir)
            except BaseException as error:
                cleanup_errors.append(error)
            importlib.invalidate_caches()
            if cleanup_errors and body_error is None:
                raise cleanup_errors[0]


def load_reward_components(reward_model_path, device, dtype=torch.bfloat16):
    """Load a trust-remote-code reward model without touching the user cache."""
    from transformers import AutoModel, AutoTokenizer

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if torch.device(device).type == "cuda":
        model_kwargs["device_map"] = {"": device}
    with temporary_hf_modules_cache():
        reward_model = AutoModel.from_pretrained(
            reward_model_path, **model_kwargs
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path, trust_remote_code=True
        )
    reward_model = reward_model.to(device).eval().requires_grad_(False)
    return reward_model, reward_tokenizer


def _cuda_device_index(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    parsed_device = torch.device(device)
    if parsed_device.type != "cuda":
        return None
    return (
        parsed_device.index
        if parsed_device.index is not None
        else torch.cuda.current_device()
    )


def capture_rng_state(device=None):
    """Capture a weights-only-safe snapshot of all process RNG streams."""
    numpy_state = np.random.get_state()
    cuda_device = _cuda_device_index(device)
    cuda_state = None
    if cuda_device is not None:
        cuda_state = {
            "device_index": cuda_device,
            "visible_device_count": torch.cuda.device_count(),
            "state": torch.cuda.get_rng_state(cuda_device),
        }
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.tensor(
                numpy_state[1].astype(np.int64), dtype=torch.int64
            ),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def restore_rng_state(state, device=None):
    """Restore a snapshot produced by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG state is missing fields: {sorted(missing)!r}")

    numpy_state = state["numpy"]
    numpy_keys = numpy_state["keys"]
    if not isinstance(numpy_keys, torch.Tensor):
        raise ValueError("NumPy RNG keys must be stored as a torch tensor")
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_keys.cpu().numpy().astype(np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_state = state["torch_cuda"]
    cuda_device = _cuda_device_index(device)
    if cuda_device is not None:
        if cuda_state is None:
            raise ValueError("Checkpoint has no CUDA RNG state for CUDA resume")
        device_count = torch.cuda.device_count()
        if int(cuda_state["visible_device_count"]) != device_count:
            raise ValueError(
                "Visible CUDA device count mismatch: "
                f"saved={cuda_state['visible_device_count']}, "
                f"current={device_count}"
            )
        if int(cuda_state["device_index"]) != cuda_device:
            raise ValueError(
                "CUDA RNG device mismatch: "
                f"saved={cuda_state['device_index']}, current={cuda_device}"
            )
        torch.cuda.set_rng_state(cuda_state["state"].cpu(), device=cuda_device)
    elif cuda_state is not None:
        raise ValueError("CUDA RNG state cannot be restored without CUDA")


def gather_rng_states(device=None):
    """Collect one RNG snapshot per rank; only rank zero receives the list."""
    local_state = capture_rng_state(device=device)
    if not dist.is_initialized():
        return [local_state]

    rank = dist.get_rank()
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


def restore_rng_state_for_rank(
    rng_state_by_rank, device=None, allow_missing=False
):
    """Restore this rank's snapshot and fail closed on topology drift."""
    if rng_state_by_rank is None:
        if allow_missing:
            return False
        raise ValueError(
            "Checkpoint RNG state is missing; use the explicit legacy-resume "
            "override only when non-exact continuation is acceptable."
        )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(rng_state_by_rank) != world_size:
        raise ValueError(
            "Checkpoint RNG world size mismatch: "
            f"saved={len(rng_state_by_rank)}, current={world_size}"
        )
    restore_rng_state(rng_state_by_rank[rank], device=device)
    return True


def _build_architecture_metadata(lm_config):
    """Snapshot every model setting that affects full-precision handoff."""
    return {
        field: copy.deepcopy(getattr(lm_config, field))
        for field in _ARCHITECTURE_METADATA_FIELDS
        if hasattr(lm_config, field)
    }


def build_checkpoint_metadata(args, lm_config, stage):
    """Build the exact training identity required for a safe resume."""
    data_path = _normalized_project_path(args.data_path)
    data_fingerprint = distributed_fingerprint_path(data_path)
    data_stat = os.stat(data_path)
    runtime_path = getattr(args, "tokenizer_path", project_path("model"))
    runtime_fingerprint = distributed_fingerprint_path(runtime_path)
    if (
        runtime_fingerprint["kind"] == "unresolved"
        and bool(getattr(args, "from_resume", 0))
        and not bool(getattr(args, "allow_legacy_resume", 0))
    ):
        raise ValueError(
            "Tokenizer/runtime content cannot be fingerprinted for exact resume: "
            f"{runtime_path!r}. Use a local immutable snapshot or the explicit "
            "legacy-resume override."
        )
    architecture = _build_architecture_metadata(lm_config)

    if hasattr(args, "lora_rank"):
        architecture["adapter"] = {
            "rank": int(args.lora_rank),
            "alpha": int(args.lora_alpha),
            "target_modules": sorted(set(args.lora_target_modules)),
        }

    training = {
        field: getattr(args, field)
        for field in _TRAINING_METADATA_FIELDS
        if hasattr(args, field)
    }
    provenance = {
        "dataset": {
            "path": data_path,
            "mtime_ns": data_stat.st_mtime_ns,
        },
        "runtime_path": _normalized_project_path(runtime_path),
        "output_dir": _normalized_project_path(args.save_dir),
    }
    if "reward_model_path" in training:
        reward_model_path = training.pop("reward_model_path")
        reward_fingerprint = distributed_fingerprint_path(reward_model_path)
        if (
            reward_fingerprint["kind"] == "unresolved"
            and bool(getattr(args, "from_resume", 0))
            and not bool(getattr(args, "allow_legacy_resume", 0))
        ):
            raise ValueError(
                "Reward model content cannot be fingerprinted for exact resume: "
                f"{reward_model_path!r}. Use a local immutable snapshot or the "
                "explicit legacy-resume override."
            )
        training["reward_model_fingerprint"] = reward_fingerprint
        provenance["reward_model_path"] = _normalized_project_path(
            reward_model_path
        )

    artifact_name = (
        getattr(args, "lora_name", None)
        if str(stage) == "lora"
        else getattr(args, "save_weight", None)
    )
    artifact_name = artifact_name or str(stage)

    metadata = {
        "schema_version": _CURRENT_CHECKPOINT_SCHEMA,
        "artifact_name": str(artifact_name),
        "stage": str(stage),
        "dataset": data_fingerprint,
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(getattr(args, "accumulation_steps", 1)),
        "max_seq_len": int(args.max_seq_len),
        "dtype": str(args.dtype),
        "compile": bool(getattr(args, "use_compile", 0)),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "runtime": runtime_fingerprint,
        "training": training,
        "architecture": architecture,
        "provenance": provenance,
    }
    if hasattr(args, "max_gen_len"):
        metadata["max_gen_len"] = int(args.max_gen_len)
    return metadata

# 设置检查点
def unwrap_model(model):
    """Recursively remove DDP and torch.compile wrappers from a model."""
    from torch.nn.parallel import DistributedDataParallel

    seen = set()
    while True:
        model_id = id(model)
        if model_id in seen:
            raise RuntimeError("Cyclic model wrapper chain detected")
        seen.add(model_id)

        if isinstance(model, DistributedDataParallel):
            model = model.module
            continue

        original_model = getattr(model, "_orig_mod", None)
        if original_model is not None and original_model is not model:
            model = original_model
            continue
        return model


def _remove_temporary_checkpoint(temp_path):
    try:
        os.remove(temp_path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def atomic_torch_save(data, target_path):
    """Write a checkpoint beside its target and atomically install it."""
    target_path = os.fspath(target_path)
    target_dir = os.path.dirname(os.path.abspath(target_path))
    prefix = f".{os.path.basename(target_path)}."
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=target_dir, prefix=prefix, suffix=".tmp"
    )
    os.close(file_descriptor)

    try:
        torch.save(data, temp_path)
        os.replace(temp_path, target_path)
    except PermissionError as error:
        removed = _remove_temporary_checkpoint(temp_path)
        temp_status = (
            "removed"
            if removed
            else f"retained at {temp_path!r} because cleanup failed"
        )
        raise PermissionError(
            f"Could not replace checkpoint {target_path!r}; the existing target "
            f"was preserved and the temporary file was {temp_status}."
        ) from error
    except BaseException:
        _remove_temporary_checkpoint(temp_path)
        raise


def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Model state {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _reference_sidecar_descriptor(model, save_dir, checkpoint_stem):
    base_model = unwrap_model(model)
    live_state = base_model.state_dict()
    versions = tuple(
        (key, int(value._version)) for key, value in live_state.items()
    )
    cache_key = (
        id(base_model),
        versions,
        os.path.normcase(os.path.abspath(save_dir)),
        checkpoint_stem,
    )
    cached = _REFERENCE_SIDECAR_CACHE.get(cache_key)
    if cached is not None:
        descriptor = cached["descriptor"]
        cached_path = os.path.join(save_dir, descriptor["file"])
        try:
            current_stat = os.stat(cached_path)
        except OSError:
            current_stat = None
        if current_stat is not None and (
            current_stat.st_size,
            current_stat.st_mtime_ns,
            current_stat.st_ino,
        ) == cached["stat"]:
            return descriptor

    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in live_state.items()
    }
    state_digest = _state_dict_sha256(cpu_state)
    file_name = f"{checkpoint_stem}_ref_{state_digest}.pth"
    sidecar_path = os.path.join(save_dir, file_name)

    sidecar_valid = False
    if os.path.isfile(sidecar_path):
        try:
            existing_state = torch.load(
                sidecar_path, map_location="cpu", weights_only=True
            )
            sidecar_valid = _state_dict_sha256(existing_state) == state_digest
        except Exception:
            sidecar_valid = False
    if not sidecar_valid:
        atomic_torch_save(cpu_state, sidecar_path)
        installed_state = torch.load(
            sidecar_path, map_location="cpu", weights_only=True
        )
        installed_digest = _state_dict_sha256(installed_state)
        if installed_digest != state_digest:
            raise RuntimeError(
                f"Reference sidecar repair failed for {sidecar_path!r}: "
                f"expected {state_digest!r}, found {installed_digest!r}."
            )
    descriptor = {
        "format": "model_state_sidecar_v1",
        "file": file_name,
        "state_sha256": state_digest,
    }
    installed_stat = os.stat(sidecar_path)
    _REFERENCE_SIDECAR_CACHE[cache_key] = {
        "descriptor": descriptor,
        "stat": (
            installed_stat.st_size,
            installed_stat.st_mtime_ns,
            installed_stat.st_ino,
        ),
    }
    return descriptor


def save_inference_weights(model, target_path):
    """Atomically save an unwrapped FP16 inference state on CPU."""
    state_dict = {
        key: value.detach().half().cpu()
        for key, value in unwrap_model(model).state_dict().items()
    }
    atomic_torch_save(state_dict, target_path)


def _strip_wrapper_prefixes(state_dict):
    """Normalize state saved by older DDP/torch.compile checkpoints."""
    clean_state = {}
    prefixes = ("module.", "_orig_mod.")
    for key, value in state_dict.items():
        clean_key = key
        prefix_removed = True
        while prefix_removed:
            prefix_removed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    prefix_removed = True
                    break
        if clean_key in clean_state:
            raise ValueError(
                f"Checkpoint keys collide after wrapper removal: {clean_key!r}"
            )
        clean_state[clean_key] = value
    return clean_state


def load_model_state(model, state_dict, strict=True):
    """Load current or legacy wrapped state into the recursively unwrapped model."""
    return unwrap_model(model).load_state_dict(
        _strip_wrapper_prefixes(state_dict), strict=strict
    )


def synchronize_model_state(model, src=0):
    """Broadcast parameters and persistent buffers for a non-DDP model."""
    if dist.is_initialized():
        for tensor in unwrap_model(model).state_dict().values():
            dist.broadcast(tensor, src=src)
    return model


def restore_checkpoint_model_state(
    model, checkpoint, key, checkpoint_dir=None, allow_missing=False
):
    """Restore an auxiliary model snapshot or fail closed for exact resume."""
    if key not in checkpoint:
        if allow_missing:
            return False
        raise ValueError(
            f"Checkpoint is missing required frozen model state {key!r}."
        )
    snapshot = checkpoint[key]
    if (
        isinstance(snapshot, dict)
        and snapshot.get("format") == "model_state_sidecar_v1"
    ):
        if checkpoint_dir is None:
            raise ValueError(
                f"Checkpoint directory is required to restore {key!r} sidecar."
            )
        file_name = snapshot.get("file")
        if not file_name or os.path.basename(file_name) != file_name:
            raise ValueError(f"Invalid {key!r} sidecar file name: {file_name!r}")
        sidecar_path = os.path.join(project_path(checkpoint_dir), file_name)
        if not os.path.isfile(sidecar_path):
            raise ValueError(f"Required {key!r} sidecar is missing: {sidecar_path!r}")
        state_dict = torch.load(
            sidecar_path, map_location="cpu", weights_only=True
        )
        actual_digest = _state_dict_sha256(state_dict)
        if actual_digest != snapshot.get("state_sha256"):
            raise ValueError(
                f"{key!r} sidecar digest mismatch: expected "
                f"{snapshot.get('state_sha256')!r}, found {actual_digest!r}."
            )
    else:
        state_dict = snapshot
    load_model_state(model, state_dict)
    return True


def _metadata_identity(metadata):
    return {
        key: metadata[key]
        for key in _RESUME_IDENTITY_FIELDS
        if key in metadata
    }


def _validate_shared_metadata_fields(saved, expected, path="metadata"):
    for key in _RESUME_IDENTITY_FIELDS:
        if key not in saved or key not in expected:
            continue
        _validate_shared_metadata_value(
            saved[key], expected[key], f"{path}.{key}"
        )


def _validate_shared_metadata_value(saved, expected, path):
    if isinstance(saved, dict) and isinstance(expected, dict):
        for key in sorted(saved.keys() & expected.keys()):
            _validate_shared_metadata_value(
                saved[key], expected[key], f"{path}.{key}"
            )
        return
    if saved != expected:
        raise ValueError(
            f"Checkpoint metadata mismatch at {path}: "
            f"expected {expected!r}, found {saved!r}."
        )


def _validate_resume_metadata(checkpoint, expected_metadata, allow_legacy_resume):
    if expected_metadata is None:
        return
    if "metadata" not in checkpoint:
        if allow_legacy_resume:
            return
        raise ValueError(
            "Checkpoint metadata is missing; set allow_legacy_resume=True only "
            "when intentionally resuming a legacy checkpoint."
        )
    saved_metadata = checkpoint["metadata"]
    if saved_metadata == expected_metadata:
        return

    saved_schema = saved_metadata.get("schema_version")
    expected_schema = expected_metadata.get("schema_version")
    if expected_schema is None:
        raise ValueError(
            "Checkpoint metadata mismatch: "
            f"expected {expected_metadata!r}, found {saved_metadata!r}."
        )
    if saved_schema is None:
        raise ValueError(
            "Checkpoint metadata has no schema_version and cannot be treated "
            "as an explicitly supported legacy schema."
        )
    if saved_schema > expected_schema:
        raise ValueError(
            f"Checkpoint uses unknown future metadata schema {saved_schema}; "
            f"this runtime supports schema {expected_schema}."
        )
    if saved_schema == expected_schema:
        if _metadata_identity(saved_metadata) == _metadata_identity(expected_metadata):
            return
        raise ValueError(
            "Checkpoint metadata mismatch: "
            f"expected {_metadata_identity(expected_metadata)!r}, "
            f"found {_metadata_identity(saved_metadata)!r}."
        )
    if (
        allow_legacy_resume
        and saved_schema in _SUPPORTED_LEGACY_METADATA_SCHEMAS
        and saved_schema < expected_schema
    ):
        _validate_shared_metadata_fields(saved_metadata, expected_metadata)
        return
    raise ValueError(
        f"Checkpoint metadata schema {saved_schema} is not accepted; expected "
        f"schema {expected_schema}."
    )


def _resume_handoff_identity_error(
    metadata, from_weight, expected_architecture, expected_runtime, legacy=False
):
    artifact_name = metadata.get("artifact_name")
    if (not legacy and artifact_name != from_weight) or (
        legacy and artifact_name is not None and artifact_name != from_weight
    ):
        prefix = "legacy " if legacy else ""
        return (
            f"{prefix}resume artifact_name {artifact_name!r} does not match "
            f"requested artifact {from_weight!r}"
        )

    expected_stage = _KNOWN_ARTIFACT_STAGES.get(from_weight)
    saved_stage = metadata.get("stage")
    if expected_stage is not None and (
        (not legacy and saved_stage != expected_stage)
        or (legacy and saved_stage is not None and saved_stage != expected_stage)
    ):
        prefix = "legacy " if legacy else ""
        return (
            f"{prefix}resume stage {saved_stage!r} does not match standard "
            f"artifact {from_weight!r} stage {expected_stage!r}"
        )

    if not legacy:
        if metadata.get("architecture") != expected_architecture:
            return (
                "resume architecture does not match the current model config: "
                f"saved={metadata.get('architecture')!r}, "
                f"current={expected_architecture!r}"
            )
        if expected_runtime.get("kind") == "unresolved":
            return "current tokenizer/runtime content cannot be fingerprinted"
        if metadata.get("runtime") != expected_runtime:
            return (
                "resume tokenizer/runtime fingerprint does not match current "
                f"content: saved={metadata.get('runtime')!r}, "
                f"current={expected_runtime!r}"
            )
        return None

    try:
        if "architecture" in metadata:
            _validate_shared_metadata_value(
                metadata["architecture"],
                expected_architecture,
                "metadata.architecture",
            )
        if "runtime" in metadata:
            if expected_runtime.get("kind") == "unresolved":
                return "current tokenizer/runtime content cannot be fingerprinted"
            _validate_shared_metadata_value(
                metadata["runtime"], expected_runtime, "metadata.runtime"
            )
    except ValueError as error:
        return f"legacy resume {error}"
    return None


def lm_checkpoint(
    lm_config,
    weight=None,
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir=None,
    metadata=None,
    expected_metadata=None,
    allow_legacy_resume=False,
    save_inference=True,
    **kwargs,
):
    if save_dir is None:
        save_dir = str(CHECKPOINT_DIR)
    else:
        save_dir = project_path(save_dir)
    # 确保保存目录存在，不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    # 构建文件名后缀：MoE 模型加 "_moe"，普通模型不加
    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    # ckp_path：只保存模型权重（半精度），用于推理加载
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    # resume_path：保存完整训练状态（模型+优化器+进度），用于断点续训
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    # ================================================================
    # 保存模式（model 不为 None）
    # ================================================================
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        
        # DDP 模型有额外封装，真正的模型在 .module 里
        # 非 DDP 模型直接调用 state_dict()
        state_dict = unwrap_model(model).state_dict()

        # ── 保存推理用的模型权重（半精度）────────────────────────────
        # 先写到 .tmp 临时文件，写完再用 os.replace 原子替换
        # 目的：防止写到一半程序崩溃导致文件损坏
        # os.replace 是原子操作，不会出现"替换到一半"的中间状态
        # half()：float32 → float16，文件大小减半，推理时精度损失可接受
        inference_state = (
            {key: value.half() for key, value in state_dict.items()}
            if save_inference
            else None
        )

        # ── 获取 wandb 实验 id（用于续训时恢复到同一个实验）──────────
        wandb_id = None
        if wandb:
            # SwanLab 的 API：通过 get_run() 获取当前实验对象
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                # WandB 的 API：直接从 wandb 对象取 id
                wandb_id = getattr(wandb, "id", None)

        # ── 构建完整训练状态字典 ─────────────────────────────────────
        resume_data = {
            # 模型权重（float32，用于续训）
            "model": state_dict,
            # 优化器状态（动量m、方差v等）
            "optimizer": optimizer.state_dict(),
            # 当前 epoch 编号
            "epoch": epoch,
            # 当前 step 编号
            "step": step,
            # 保存当前 GPU 数量，续训时 GPU 数量变化时用于换算 step
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            # 实验 id，续训时恢复实验曲线
            "wandb_id": wandb_id,
        }
        if metadata is not None:
            resume_data["metadata"] = metadata

        reference_model = kwargs.pop("ref_model", None)
        if reference_model is not None:
            checkpoint_stem = (
                f"{weight}_{lm_config.hidden_size}{moe_path}"
            )
            resume_data["ref_model"] = _reference_sidecar_descriptor(
                reference_model, save_dir, checkpoint_stem
            )

        # ── 处理额外的可选状态（如 scaler）──────────────────────────
        # 调用时传入 scaler=scaler，kwargs = {"scaler": scaler}
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    # 有 state_dict 方法的对象（如 GradScaler、模型）
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = unwrap_model(value).state_dict()
                    else:
                        # scaler → resume_data["scaler"] = scaler.state_dict()
                        resume_data[key] = unwrap_model(value).state_dict()
                else:
                    # 普通值（如整数、字符串）直接存
                    resume_data[key] = value
        # ── 保存完整训练状态（同样用 tmp + replace 原子写入）─────────
        atomic_torch_save(resume_data, resume_path)
        # 先安装权威的全精度训练状态。即使半精度推理文件被占用，最新的
        # 可续训状态仍然可用，跨阶段加载也不会退回到旧的 resume 文件。
        if save_inference:
            atomic_torch_save(inference_state, ckp_path)
        # Windows 上 os.replace 的限制：目标文件被其他进程占用时，Windows 不允许替换，Linux 没有这个问题
        # os.replace(resume_tmp, resume_path)

    # ================================================================
    # 加载模式（model 为 None，只传了 lm_config 和 weight）
    # ================================================================
    else:
        # resume_path 存在 → 有上次的训练状态，可以续训
        if os.path.exists(resume_path):
            # map_location="cpu"：先加载到 CPU，避免 GPU 显存直接被占用
            # 后续训练循环会手动把参数移到对应设备
            ckp_data = torch.load(resume_path, map_location="cpu", weights_only=True)
            # ── 处理 GPU 数量变化的情况 ──────────────────────────────
            # 上次用 4 张卡训练，这次只有 2 张卡
            # 每张卡处理的数据量不同，step 编号需要换算
            saved_ws = ckp_data.get("world_size", 1) # 上次训练的 GPU 数量
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                raise ValueError(
                    "Checkpoint world_size mismatch: "
                    f"saved={saved_ws}, current={current_ws}. Resume with the "
                    "same world size instead of rescaling the saved step."
                )
            # 返回完整训练状态，调用方用它恢复 model/optimizer/scaler/epoch/step
            _validate_resume_metadata(
                ckp_data, expected_metadata, allow_legacy_resume
            )
            return ckp_data
        # resume_path 不存在 → 没有checkpoint，返回 None，从头开始训练
        return None

# 初始化模型
def init_model(
    lm_config,
    from_weight=None,
    tokenizer_path=None,
    lora_weight="None", 
    save_dir=None,
    resume_dir=None,
    device="cuda",
    allow_legacy_resume=False,
):
    from transformers import AutoTokenizer
    from model.MiniMindModel import MiniMindForCausalLM

    # 如果没有指定 tokenizer_path，使用项目根目录下的 model 文件夹
    if tokenizer_path is None:
        # 获取当前文件所在目录的父目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    expected_architecture = _build_architecture_metadata(lm_config)
    expected_runtime = fingerprint_path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    save_dir = project_path("out" if save_dir is None else save_dir)

    model = MiniMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = (
            "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        )
        checkpoint_name = (
            f"{from_weight}_{lm_config.hidden_size}{moe_suffix}"
        )
        resume_path = (
            os.path.join(
                project_path(resume_dir), f"{checkpoint_name}_resume.pth"
            )
            if resume_dir is not None
            else None
        )
        weight_path = os.path.join(save_dir, f"{checkpoint_name}.pth")

        resume_exists = resume_path is not None and os.path.isfile(resume_path)
        weight_exists = os.path.isfile(weight_path)
        use_resume = False
        resume_error = None
        resume_data = None
        if resume_exists:
            resume_data = torch.load(
                resume_path, map_location="cpu", weights_only=True
            )
            metadata = resume_data.get("metadata")
            if metadata is None:
                if allow_legacy_resume:
                    use_resume = True
                else:
                    resume_error = (
                        "resume checkpoint has no metadata; pass "
                        "allow_legacy_resume=True only for an intentional "
                        "legacy handoff"
                    )
            else:
                schema = metadata.get("schema_version")
                if schema == _CURRENT_CHECKPOINT_SCHEMA:
                    resume_error = _resume_handoff_identity_error(
                        metadata,
                        from_weight,
                        expected_architecture,
                        expected_runtime,
                    )
                    if resume_error is None:
                        use_resume = True
                elif schema is not None and schema > _CURRENT_CHECKPOINT_SCHEMA:
                    resume_error = (
                        f"resume checkpoint uses unknown future schema {schema}"
                    )
                elif (
                    allow_legacy_resume
                    and schema in _SUPPORTED_LEGACY_METADATA_SCHEMAS
                ):
                    resume_error = _resume_handoff_identity_error(
                        metadata,
                        from_weight,
                        expected_architecture,
                        expected_runtime,
                        legacy=True,
                    )
                    if resume_error is None:
                        use_resume = True
                else:
                    resume_error = (
                        f"resume checkpoint schema {schema!r} is not a supported "
                        "scoped handoff; pass allow_legacy_resume=True only for "
                        "a supported legacy schema"
                    )

            if use_resume and "model" not in resume_data:
                use_resume = False
                resume_error = "resume checkpoint has no model state"

        if use_resume:
            weights = resume_data["model"]
        elif resume_exists:
            raise ValueError(
                f"Resume checkpoint {resume_path!r} is not a valid handoff "
                f"for artifact {from_weight!r}: {resume_error}. Refusing to "
                "fall back to inference weights without validated identity "
                "metadata."
            )
        elif weight_exists:
            weights = torch.load(
                weight_path, map_location="cpu", weights_only=True
            )
        else:
            detail = f" Resume was unusable: {resume_error}." if resume_error else ""
            raise FileNotFoundError(
                f"No usable model source for artifact {from_weight!r}: "
                f"inference weights were not found at {weight_path!r}." + detail
            )

        model.load_state_dict(_strip_wrapper_prefixes(weights), strict=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer


def init_reference_model(
    lm_config,
    from_weight,
    checkpoint=None,
    tokenizer_path=None,
    save_dir=None,
    checkpoint_dir=None,
    device="cuda",
    allow_legacy_resume=False,
):
    """Build a frozen-model candidate directly from its exact resume snapshot."""
    from model.MiniMindModel import MiniMindForCausalLM

    def load_primary_reference():
        if checkpoint is not None and "ref_model" in checkpoint:
            reference_model = MiniMindForCausalLM(lm_config)
            restore_checkpoint_model_state(
                reference_model,
                checkpoint,
                key="ref_model",
                checkpoint_dir=checkpoint_dir,
            )
            return reference_model.to(device), True

        if checkpoint is not None and not allow_legacy_resume:
            raise ValueError(
                "Checkpoint is missing required frozen model state 'ref_model'."
            )

        reference_model, _ = init_model(
            lm_config,
            from_weight=from_weight,
            tokenizer_path=tokenizer_path,
            save_dir=save_dir,
            resume_dir=checkpoint_dir,
            device=device,
            allow_legacy_resume=allow_legacy_resume,
        )
        return reference_model, False

    if not dist.is_initialized():
        return load_primary_reference()

    rank = dist.get_rank()
    status = [None]
    reference_model = None
    rank_zero_error = None
    if rank == 0:
        try:
            reference_model, restored = load_primary_reference()
            status[0] = {"restored": restored}
        except BaseException as error:
            rank_zero_error = error
            status[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(status, src=0)
    if "error" in status[0]:
        if rank == 0 and not isinstance(rank_zero_error, Exception):
            raise rank_zero_error
        raise RuntimeError(
            f"Could not initialize frozen reference model: {status[0]['error']}"
        )
    if rank != 0:
        reference_model = MiniMindForCausalLM(lm_config).to(device)
    return reference_model, bool(status[0]["restored"])


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        # sampler：底层采样器，决定数据的遍历顺序
        #   - 分布式训练时是 DistributedSampler（确保不同GPU拿不同数据）
        #   - 单卡训练时是 indices（torch.randperm 生成的随机索引列表）
        self.sampler = sampler
        # 每个 batch 包含多少条样本
        self.batch_size = batch_size
        # 要跳过的 batch 数量，对应上次已训练的 step 数
        # skip_batches=0 表示从头开始，不跳过任何数据
        self.skip_batches = skip_batches

    def __iter__(self):
        # 当前正在积累的 batch，存放样本索引
        batch = []
        # 已经跳过的 batch 数量
        skipped = 0

        for idx in self.sampler:
            # 把当前样本索引加入 batch
            batch.append(idx)
            # batch 积累满了
            if len(batch) == self.batch_size:
                # 还没跳够，丢弃这个 batch
                if skipped < self.skip_batches:
                    # 跳过计数 +1
                    skipped += 1
                    # 清空，重新积累下一个 batch
                    batch = []
                    # 继续遍历下一个 idx
                    continue

                # 已跳过足够的 batch，正常产出
                yield batch
                # 清空，准备积累下一个 batch
                batch = []
                
        # 处理最后一个不完整的 batch（样本数 < batch_size）
        # 必须满足 skipped >= skip_batches，即跳过阶段已经结束
        # 如果跳过阶段还没结束，说明整个数据集都被跳过了，不产出任何数据
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        return max(0, total_batches - self.skip_batches)


class LengthBucketBatchSampler(Sampler):
    """Batch a deterministic index stream after sorting bounded local windows."""

    def __init__(
        self,
        sampler,
        batch_size,
        lengths,
        skip_batches=0,
        bucket_window_multiplier=50,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.lengths = lengths
        self.skip_batches = skip_batches
        self.bucket_window_size = batch_size * max(1, bucket_window_multiplier)

    def __iter__(self):
        window = []
        skipped = 0

        for index in self.sampler:
            window.append(index)
            if len(window) < self.bucket_window_size:
                continue

            sorted_window = sorted(
                window, key=lambda sample_index: self.lengths[sample_index]
            )
            for batch_start in range(0, len(sorted_window), self.batch_size):
                batch = sorted_window[batch_start:batch_start + self.batch_size]
                if skipped < self.skip_batches:
                    skipped += 1
                else:
                    yield batch
            window = []

        if window:
            sorted_window = sorted(
                window, key=lambda sample_index: self.lengths[sample_index]
            )
            for batch_start in range(0, len(sorted_window), self.batch_size):
                batch = sorted_window[batch_start:batch_start + self.batch_size]
                if skipped < self.skip_batches:
                    skipped += 1
                else:
                    yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def build_epoch_batch_sampler(
    dataset_size,
    batch_size,
    epoch,
    skip_batches=0,
    sampler=None,
    seed=42,
    lengths=None,
    bucket_window_multiplier=50,
):
    """Build one deterministic epoch batch stream for fresh and resumed training."""
    if sampler is None:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        sampler = torch.randperm(dataset_size, generator=generator).tolist()
    if lengths is not None:
        return LengthBucketBatchSampler(
            sampler,
            batch_size,
            lengths,
            skip_batches=skip_batches,
            bucket_window_multiplier=bucket_window_multiplier,
        )
    return SkipBatchSampler(sampler, batch_size, skip_batches)
