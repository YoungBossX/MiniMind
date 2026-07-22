import ast
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from trainer import trainer_utils


TRAINER_PATHS = (
    "trainer/train_pretrain.py",
    "trainer/train_full_sft.py",
    "trainer/train_lora.py",
    "trainer/train_dpo.py",
    "trainer/train_reason.py",
    "trainer/train_ppo.py",
    "trainer/train_grpo.py",
)

FULL_WEIGHT_TRAINER_PATHS = tuple(
    path for path in TRAINER_PATHS if path != "trainer/train_lora.py"
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


class _CompiledWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._orig_mod = model


def _config():
    return SimpleNamespace(hidden_size=2, use_moe=False)


def _argument_default(relative_path, option):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
        return None
    raise AssertionError(f"{relative_path} does not define {option}")


def _checkpoint_calls(relative_path):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "lm_checkpoint"
    ]


def _loads_primary_model_through_safe_loader(relative_path):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "load_model_state" or len(node.args) < 2:
            continue
        state = node.args[1]
        if not (
            isinstance(state, ast.Subscript)
            and isinstance(state.value, ast.Name)
            and state.value.id == "ckp_data"
        ):
            continue
        return True
    return False


def _fake_ddp(model):
    wrapper = DistributedDataParallel.__new__(DistributedDataParallel)
    nn.Module.__init__(wrapper)
    wrapper.module = model
    return wrapper


def _save_tiny_checkpoint(save_dir, weight="unit", metadata=None, **extra):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer_utils.lm_checkpoint(
        _config(),
        weight=weight,
        model=model,
        optimizer=optimizer,
        epoch=3,
        step=11,
        save_dir=str(save_dir),
        metadata=metadata,
        **extra,
    )
    return model


def test_build_checkpoint_metadata_captures_training_and_data_identity(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "one"}\n', encoding="utf-8")
    args = SimpleNamespace(
        data_path=str(data_path),
        save_dir=str(tmp_path / "out"),
        batch_size=4,
        accumulation_steps=8,
        max_seq_len=512,
        dtype="bfloat16",
        use_compile=1,
        epochs=2,
        learning_rate=1e-6,
        grad_clip=1.0,
        num_workers=2,
        from_weight="pretrain",
        save_weight="full_sft",
    )
    config = SimpleNamespace(
        model_type="minimind",
        hidden_size=512,
        intermediate_size=1408,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=6400,
        max_position_embeddings=32768,
        use_moe=False,
    )

    metadata = trainer_utils.build_checkpoint_metadata(args, config, stage="sft")
    stat = data_path.stat()

    assert metadata["schema_version"] == 2
    assert metadata["stage"] == "sft"
    assert metadata["artifact_name"] == "full_sft"
    dataset_identity = dict(metadata["dataset"])
    dataset_sha256 = dataset_identity.pop("sha256")
    assert len(dataset_sha256) == 64
    assert dataset_identity == {
        "kind": "file",
        "file_count": 1,
        "size_bytes": stat.st_size,
    }
    assert metadata["batch_size"] == 4
    assert metadata["accumulation_steps"] == 8
    assert metadata["max_seq_len"] == 512
    assert metadata["dtype"] == "bfloat16"
    assert metadata["compile"] is True
    assert metadata["world_size"] == 1
    assert metadata["provenance"]["dataset"] == {
        "path": os.path.normcase(str(data_path.resolve())),
        "mtime_ns": stat.st_mtime_ns,
    }
    assert metadata["provenance"]["output_dir"] == os.path.normcase(
        str((tmp_path / "out").resolve())
    )
    assert metadata["training"] == {
        "epochs": 2,
        "learning_rate": 1e-6,
        "grad_clip": 1.0,
        "num_workers": 2,
        "from_weight": "pretrain",
    }
    assert metadata["architecture"]["hidden_size"] == 512
    assert metadata["architecture"]["num_hidden_layers"] == 8


def test_rng_state_round_trip_is_weights_only_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_utils.torch.cuda, "is_available", lambda: False)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    state = trainer_utils.capture_rng_state()
    checkpoint_path = tmp_path / "rng.pth"
    torch.save({"rng_state_by_rank": [state]}, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = (random.random(), np.random.random(), torch.rand(3))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    trainer_utils.restore_rng_state(loaded["rng_state_by_rank"][0])
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_gather_rng_states_preserves_rank_order_on_main(monkeypatch):
    local_state = {"rank": 0}
    remote_state = {"rank": 1}
    monkeypatch.setattr(
        trainer_utils, "capture_rng_state", lambda device=None: local_state
    )
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_utils.dist, "get_world_size", lambda: 2)

    def fake_gather_object(value, gathered, dst):
        assert value is local_state
        assert dst == 0
        gathered[:] = [value, remote_state]

    monkeypatch.setattr(trainer_utils.dist, "gather_object", fake_gather_object)

    assert trainer_utils.gather_rng_states() == [local_state, remote_state]


def test_rng_capture_reads_only_the_current_cuda_device(monkeypatch):
    observed_devices = []
    expected_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    monkeypatch.setattr(trainer_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_utils.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(trainer_utils.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        trainer_utils.torch.cuda,
        "get_rng_state",
        lambda device: observed_devices.append(device) or expected_state,
    )
    monkeypatch.setattr(
        trainer_utils.torch.cuda,
        "get_rng_state_all",
        lambda: pytest.fail("must not initialize every visible CUDA device"),
    )

    state = trainer_utils.capture_rng_state()

    assert observed_devices == [2]
    assert state["torch_cuda"]["device_index"] == 2
    assert state["torch_cuda"]["visible_device_count"] == 4
    assert torch.equal(state["torch_cuda"]["state"], expected_state)


def test_restore_rng_state_for_rank_validates_presence_and_world_size(
    monkeypatch,
):
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(trainer_utils.dist, "get_world_size", lambda: 2)
    restored = []
    monkeypatch.setattr(
        trainer_utils,
        "restore_rng_state",
        lambda state, device=None: restored.append(state),
    )

    assert trainer_utils.restore_rng_state_for_rank([{"rank": 0}, {"rank": 1}])
    assert restored == [{"rank": 1}]
    with pytest.raises(ValueError, match="world size"):
        trainer_utils.restore_rng_state_for_rank([{"rank": 0}])
    with pytest.raises(ValueError, match="missing"):
        trainer_utils.restore_rng_state_for_rank(None)
    assert trainer_utils.restore_rng_state_for_rank(None, allow_missing=True) is False


def test_disabled_gradient_clipping_still_rejects_nonfinite_gradients():
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([2.0])

    norm = trainer_utils.clip_gradients([parameter], max_norm=0)

    torch.testing.assert_close(norm, torch.tensor(2.0))
    torch.testing.assert_close(parameter.grad, torch.tensor([2.0]))

    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer_utils.clip_gradients([parameter], max_norm=0)


def test_restore_checkpoint_model_state_fails_closed_without_snapshot():
    reference = _TinyModel()
    expected = torch.full((2, 2), 3.0)

    assert trainer_utils.restore_checkpoint_model_state(
        reference,
        {"ref_model": {"linear.weight": expected}},
        key="ref_model",
    )
    assert torch.equal(reference.linear.weight, expected)

    with pytest.raises(ValueError, match="ref_model"):
        trainer_utils.restore_checkpoint_model_state(
            reference, {}, key="ref_model"
        )
    assert not trainer_utils.restore_checkpoint_model_state(
        reference, {}, key="ref_model", allow_missing=True
    )


def test_reference_model_checkpoint_uses_content_addressed_sidecar(tmp_path):
    first_reference = _TinyModel()
    second_reference = _TinyModel()
    with torch.no_grad():
        first_reference.linear.weight.fill_(3.0)
        second_reference.linear.weight.fill_(4.0)

    _save_tiny_checkpoint(
        tmp_path, weight="sidecar", ref_model=first_reference
    )
    first_checkpoint = trainer_utils.lm_checkpoint(
        _config(), weight="sidecar", save_dir=str(tmp_path)
    )
    first_descriptor = first_checkpoint["ref_model"]
    assert first_descriptor["format"] == "model_state_sidecar_v1"
    first_sidecar = tmp_path / first_descriptor["file"]
    assert first_sidecar.is_file()

    restored = _TinyModel()
    assert trainer_utils.restore_checkpoint_model_state(
        restored,
        first_checkpoint,
        key="ref_model",
        checkpoint_dir=str(tmp_path),
    )
    assert torch.equal(
        restored.linear.weight, first_reference.linear.weight
    )

    _save_tiny_checkpoint(
        tmp_path, weight="sidecar", ref_model=second_reference
    )
    second_checkpoint = trainer_utils.lm_checkpoint(
        _config(), weight="sidecar", save_dir=str(tmp_path)
    )
    second_descriptor = second_checkpoint["ref_model"]
    second_sidecar = tmp_path / second_descriptor["file"]

    assert second_descriptor["state_sha256"] != first_descriptor["state_sha256"]
    assert second_sidecar != first_sidecar
    assert first_sidecar.is_file()
    assert second_sidecar.is_file()


def test_init_reference_model_uses_snapshot_without_loading_mutable_base(
    monkeypatch,
):
    import model.MiniMindModel as model_module

    expected = torch.full((2, 2), 7.0)
    monkeypatch.setattr(
        model_module, "MiniMindForCausalLM", lambda _config: _TinyModel()
    )
    monkeypatch.setattr(
        trainer_utils,
        "init_model",
        lambda *_args, **_kwargs: pytest.fail(
            "exact resume must not reload the mutable base model"
        ),
    )

    reference, restored = trainer_utils.init_reference_model(
        _config(),
        from_weight="full_sft",
        checkpoint={"ref_model": {"linear.weight": expected}},
        save_dir="out",
        checkpoint_dir="checkpoints",
        device="cpu",
    )

    assert restored
    assert torch.equal(reference.linear.weight, expected)


def test_distributed_nonzero_rank_does_not_read_reference_sidecar(monkeypatch):
    import model.MiniMindModel as model_module

    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        model_module, "MiniMindForCausalLM", lambda _config: _TinyModel()
    )
    monkeypatch.setattr(
        trainer_utils,
        "restore_checkpoint_model_state",
        lambda *_args, **_kwargs: pytest.fail(
            "nonzero rank must not read the reference sidecar"
        ),
    )
    monkeypatch.setattr(
        trainer_utils,
        "init_model",
        lambda *_args, **_kwargs: pytest.fail(
            "nonzero rank must not read mutable base weights"
        ),
    )

    def fake_broadcast(status, src):
        assert src == 0
        status[0] = {"restored": True}

    monkeypatch.setattr(
        trainer_utils.dist, "broadcast_object_list", fake_broadcast
    )

    reference, restored = trainer_utils.init_reference_model(
        _config(),
        from_weight="full_sft",
        checkpoint={"ref_model": {"linear.weight": torch.ones(2, 2)}},
        save_dir="out",
        checkpoint_dir="checkpoints",
        device="cpu",
    )

    assert restored
    assert isinstance(reference, _TinyModel)


def test_init_reference_model_broadcasts_base_exception_before_rank_zero_reraises(
    monkeypatch,
):
    import model.MiniMindModel as model_module

    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        model_module, "MiniMindForCausalLM", lambda _config: _TinyModel()
    )
    monkeypatch.setattr(
        trainer_utils,
        "restore_checkpoint_model_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("reference stopped")
        ),
    )
    broadcasts = []
    monkeypatch.setattr(
        trainer_utils.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((dict(status[0]), src)),
    )

    with pytest.raises(KeyboardInterrupt, match="reference stopped"):
        trainer_utils.init_reference_model(
            _config(),
            from_weight="full_sft",
            checkpoint={"ref_model": {"linear.weight": torch.ones(2, 2)}},
            checkpoint_dir="checkpoints",
            device="cpu",
        )

    assert broadcasts == [
        ({"error": "KeyboardInterrupt: reference stopped"}, 0)
    ]


def test_init_reference_model_nonzero_rank_raises_broadcast_base_exception(
    monkeypatch,
):
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 1)

    def broadcast(status, src):
        assert status == [None]
        assert src == 0
        status[0] = {"error": "SystemExit: reference stopped"}

    monkeypatch.setattr(trainer_utils.dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="SystemExit: reference stopped"):
        trainer_utils.init_reference_model(
            _config(),
            from_weight="full_sft",
            checkpoint={"ref_model": {"linear.weight": torch.ones(2, 2)}},
            checkpoint_dir="checkpoints",
            device="cpu",
        )


def test_synchronize_model_state_broadcasts_parameters_and_buffers(monkeypatch):
    model = _TinyModel()
    model.register_buffer("marker", torch.tensor([1.0]))
    observed = []
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        trainer_utils.dist,
        "broadcast",
        lambda tensor, src: observed.append((tensor, src)),
    )

    assert trainer_utils.synchronize_model_state(model) is model
    assert len(observed) == len(model.state_dict())
    assert all(src == 0 for _tensor, src in observed)


def test_checkpoint_metadata_changes_when_dataset_file_changes(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "one"}\n', encoding="utf-8")
    args = SimpleNamespace(
        data_path=str(data_path),
        save_dir=str(tmp_path / "out"),
        batch_size=1,
        accumulation_steps=1,
        max_seq_len=32,
        dtype="bfloat16",
        use_compile=0,
    )

    before = trainer_utils.build_checkpoint_metadata(args, _config(), "pretrain")
    with data_path.open("a", encoding="utf-8") as handle:
        handle.write('{"text": "two"}\n')
    after = trainer_utils.build_checkpoint_metadata(args, _config(), "pretrain")

    assert before["dataset"] != after["dataset"]


def test_dataset_fingerprint_detects_same_size_preserved_mtime_replacement(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "one"}\n', encoding="utf-8")
    original_stat = data_path.stat()
    args = SimpleNamespace(
        data_path=str(data_path),
        save_dir=str(tmp_path / "out"),
        batch_size=1,
        accumulation_steps=1,
        max_seq_len=32,
        dtype="bfloat16",
        use_compile=0,
    )

    before = trainer_utils.build_checkpoint_metadata(args, _config(), "pretrain")
    data_path.write_text('{"text": "two"}\n', encoding="utf-8")
    os.utime(
        data_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after = trainer_utils.build_checkpoint_metadata(args, _config(), "pretrain")

    assert before["dataset"]["size_bytes"] == after["dataset"]["size_bytes"]
    assert (
        before["provenance"]["dataset"]["mtime_ns"]
        == after["provenance"]["dataset"]["mtime_ns"]
    )
    assert before["dataset"]["sha256"] != after["dataset"]["sha256"]


def test_single_file_fingerprint_ignores_physical_filename(tmp_path):
    first = tmp_path / "node-a" / "train-a.jsonl"
    second = tmp_path / "node-b" / "renamed.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("identical content\n", encoding="utf-8")
    second.write_text("identical content\n", encoding="utf-8")

    assert trainer_utils.fingerprint_path(first) == trainer_utils.fingerprint_path(
        second
    )


def test_metadata_identity_accepts_renamed_single_file_mount(tmp_path):
    first = tmp_path / "node-a" / "train-a.jsonl"
    second = tmp_path / "node-b" / "renamed.jsonl"
    runtime = tmp_path / "runtime"
    first.parent.mkdir()
    second.parent.mkdir()
    runtime.mkdir()
    first.write_text("identical content\n", encoding="utf-8")
    second.write_text("identical content\n", encoding="utf-8")
    (runtime / "tokenizer.json").write_text("tokenizer\n", encoding="utf-8")

    first_metadata = trainer_utils.build_checkpoint_metadata(
        _metadata_args(first, runtime, tmp_path / "out-a"), _config(), "dpo"
    )
    second_metadata = trainer_utils.build_checkpoint_metadata(
        _metadata_args(second, runtime, tmp_path / "out-b"), _config(), "dpo"
    )
    checkpoint_dir = tmp_path / "checkpoints"
    _save_tiny_checkpoint(
        checkpoint_dir, weight="renamed-data", metadata=first_metadata
    )

    resumed = trainer_utils.lm_checkpoint(
        _config(),
        weight="renamed-data",
        save_dir=str(checkpoint_dir),
        expected_metadata=second_metadata,
    )
    assert resumed["metadata"] == first_metadata


def test_runtime_fingerprint_detects_tokenizer_asset_changes(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "one"}\n', encoding="utf-8")
    runtime_path = tmp_path / "model-runtime"
    runtime_path.mkdir()
    tokenizer_path = runtime_path / "tokenizer.json"
    tokenizer_path.write_text('{"version": 1}', encoding="utf-8")
    args = SimpleNamespace(
        data_path=str(data_path),
        tokenizer_path=str(runtime_path),
        save_dir=str(tmp_path / "out"),
        batch_size=1,
        accumulation_steps=1,
        max_seq_len=32,
        dtype="bfloat16",
        use_compile=0,
    )

    before = trainer_utils.build_checkpoint_metadata(args, _config(), "pretrain")
    tokenizer_path.write_text('{"version": 2}', encoding="utf-8")
    after = trainer_utils.build_checkpoint_metadata(args, _config(), "pretrain")

    assert before["runtime"]["sha256"] != after["runtime"]["sha256"]


def test_training_hf_module_cache_is_workspace_local_and_self_cleaning(tmp_path):
    from transformers import dynamic_module_utils

    original_cache = dynamic_module_utils.HF_MODULES_CACHE
    with trainer_utils.temporary_hf_modules_cache(tmp_path) as cache_path:
        cache_path = Path(cache_path)
        assert cache_path.parent.resolve() == tmp_path.resolve()
        assert Path(dynamic_module_utils.HF_MODULES_CACHE).resolve() == cache_path.resolve()
        (cache_path / "sentinel.py").write_text("# temp\n", encoding="utf-8")

    assert dynamic_module_utils.HF_MODULES_CACHE == original_cache
    assert not cache_path.exists()


def test_rl_checkpoint_metadata_captures_scheduler_and_policy_identity(tmp_path):
    data_path = tmp_path / "rl.jsonl"
    data_path.write_text('{"prompt": "question"}\n', encoding="utf-8")
    reward_path = tmp_path / "reward-model"
    reward_path.mkdir()
    args = SimpleNamespace(
        data_path=str(data_path),
        save_dir=str(tmp_path / "run-a"),
        batch_size=2,
        accumulation_steps=1,
        max_seq_len=66,
        max_gen_len=512,
        dtype="bfloat16",
        use_compile=0,
        epochs=3,
        learning_rate=1e-6,
        critic_learning_rate=2e-6,
        temperature=0.8,
        reasoning=1,
        reward_model_path=str(reward_path),
        num_generations=4,
        grpo_epochs=2,
        ppo_epochs=4,
        gamma=1.0,
        gae_lambda=0.95,
        beta=0.02,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        kl_coef=0.02,
        vf_coef=0.5,
        target_kl=0.02,
        grad_clip=1.0,
    )

    metadata = trainer_utils.build_checkpoint_metadata(args, _config(), "ppo")

    assert metadata["max_gen_len"] == 512
    training = dict(metadata["training"])
    reward_fingerprint = training.pop("reward_model_fingerprint")
    assert reward_fingerprint["kind"] == "directory"
    assert reward_fingerprint["file_count"] == 0
    assert reward_fingerprint["size_bytes"] == 0
    assert len(reward_fingerprint["sha256"]) == 64
    assert training == {
        "epochs": 3,
        "learning_rate": 1e-6,
        "critic_learning_rate": 2e-6,
        "temperature": 0.8,
        "reasoning": 1,
        "num_generations": 4,
        "grpo_epochs": 2,
        "ppo_epochs": 4,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "beta": 0.02,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "kl_coef": 0.02,
        "vf_coef": 0.5,
        "target_kl": 0.02,
        "grad_clip": 1.0,
    }
    assert metadata["provenance"]["reward_model_path"] == os.path.normcase(
        str(reward_path.resolve())
    )


def test_reward_model_fingerprint_changes_when_contents_change(tmp_path):
    data_path = tmp_path / "rl.jsonl"
    data_path.write_text('{"prompt": "question"}\n', encoding="utf-8")
    reward_path = tmp_path / "reward-model"
    reward_path.mkdir()
    weight_path = reward_path / "weights.bin"
    weight_path.write_bytes(b"first")
    args = SimpleNamespace(
        data_path=str(data_path),
        save_dir=str(tmp_path / "out"),
        batch_size=1,
        accumulation_steps=1,
        max_seq_len=32,
        dtype="bfloat16",
        use_compile=0,
        reward_model_path=str(reward_path),
    )

    before = trainer_utils.build_checkpoint_metadata(args, _config(), "ppo")
    weight_path.write_bytes(b"other")
    after = trainer_utils.build_checkpoint_metadata(args, _config(), "ppo")

    assert (
        before["training"]["reward_model_fingerprint"]
        != after["training"]["reward_model_fingerprint"]
    )


def test_reward_model_fingerprint_ignores_non_runtime_reports(tmp_path):
    reward_path = tmp_path / "reward-model"
    reward_path.mkdir()
    (reward_path / "config.json").write_text("{}", encoding="utf-8")
    (reward_path / "model.safetensors").write_bytes(b"weights")
    (reward_path / "README.md").write_text("first", encoding="utf-8")
    results_dir = reward_path / "reward_bench_results"
    results_dir.mkdir()
    result_path = results_dir / "results.json"
    result_path.write_text('{"score": 1}', encoding="utf-8")

    before = trainer_utils.fingerprint_path(reward_path)
    (reward_path / "README.md").write_text("second", encoding="utf-8")
    result_path.write_text('{"score": 2}', encoding="utf-8")
    after = trainer_utils.fingerprint_path(reward_path)

    assert after == before


def test_reward_model_fingerprint_includes_nested_runtime_assets(tmp_path):
    reward_path = tmp_path / "reward-model"
    nested_path = reward_path / "custom_package"
    nested_path.mkdir(parents=True)
    runtime_asset = nested_path / "chat_template.jinja"
    runtime_asset.write_text("first", encoding="utf-8")

    before = trainer_utils.fingerprint_path(reward_path)
    runtime_asset.write_text("second", encoding="utf-8")
    after = trainer_utils.fingerprint_path(reward_path)

    assert after != before


def test_model_fingerprint_follows_directory_symlink_content(tmp_path):
    runtime_path = tmp_path / "runtime"
    runtime_path.mkdir()
    first_target = tmp_path / "target-a"
    second_target = tmp_path / "target-b"
    first_target.mkdir()
    second_target.mkdir()
    (first_target / "asset.bin").write_bytes(b"first")
    (second_target / "asset.bin").write_bytes(b"second")
    link_path = runtime_path / "custom_package"
    try:
        link_path.symlink_to(first_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    before = trainer_utils.fingerprint_path(runtime_path)
    link_path.unlink()
    link_path.symlink_to(second_target, target_is_directory=True)
    after = trainer_utils.fingerprint_path(runtime_path)

    assert after != before


def test_model_fingerprint_propagates_nested_permission_errors(tmp_path, monkeypatch):
    runtime_path = tmp_path / "runtime"
    blocked_path = runtime_path / "blocked"
    blocked_path.mkdir(parents=True)
    (blocked_path / "asset.bin").write_bytes(b"content")
    original_scandir = os.scandir

    def guarded_scandir(path):
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
            os.path.abspath(blocked_path)
        ):
            raise PermissionError("blocked for test")
        return original_scandir(path)

    monkeypatch.setattr(trainer_utils.os, "scandir", guarded_scandir)

    with pytest.raises(PermissionError, match="blocked for test"):
        trainer_utils.fingerprint_path(runtime_path)


def test_distributed_fingerprint_reads_content_only_on_rank_zero(monkeypatch):
    expected = {"kind": "directory", "sha256": "abc"}
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(trainer_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        trainer_utils,
        "fingerprint_path",
        lambda _path: pytest.fail("nonzero rank must not hash shared files"),
    )

    calls = []

    def fake_all_gather(gathered, payload):
        calls.append(payload)
        if len(calls) == 1:
            gathered[:] = ["node-a", "node-a"]
        else:
            assert payload is None
            gathered[:] = [{"fingerprint": expected}, None]

    monkeypatch.setattr(
        trainer_utils.dist, "all_gather_object", fake_all_gather
    )

    assert trainer_utils.distributed_fingerprint_path("reward") == expected


def test_distributed_fingerprint_rejects_different_node_snapshots(monkeypatch):
    local = {"kind": "directory", "sha256": "node-a"}
    remote = {"kind": "directory", "sha256": "node-b"}
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(trainer_utils, "fingerprint_path", lambda _path: local)

    calls = []

    def fake_all_gather(gathered, payload):
        calls.append(payload)
        if len(calls) == 1:
            gathered[:] = ["node-a", "node-b"]
        else:
            assert payload == {"fingerprint": local}
            gathered[:] = [payload, {"fingerprint": remote}]

    monkeypatch.setattr(
        trainer_utils.dist, "all_gather_object", fake_all_gather
    )

    with pytest.raises(ValueError, match="different content"):
        trainer_utils.distributed_fingerprint_path("reward")


def test_unresolved_reward_model_identity_requires_explicit_legacy_resume(tmp_path):
    data_path = tmp_path / "rl.jsonl"
    data_path.write_text('{"prompt": "question"}\n', encoding="utf-8")
    args = SimpleNamespace(
        data_path=str(data_path),
        save_dir=str(tmp_path / "out"),
        batch_size=1,
        accumulation_steps=1,
        max_seq_len=32,
        dtype="bfloat16",
        use_compile=0,
        reward_model_path="hf-org/definitely-missing-reward",
        from_resume=1,
        allow_legacy_resume=0,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        trainer_utils.build_checkpoint_metadata(args, _config(), "ppo")

    args.allow_legacy_resume = 1
    metadata = trainer_utils.build_checkpoint_metadata(args, _config(), "ppo")
    assert metadata["training"]["reward_model_fingerprint"]["kind"] == "unresolved"


def test_checkpoint_metadata_captures_stage_specific_objective_identity(tmp_path):
    data_path = tmp_path / "stage.jsonl"
    data_path.write_text('{"text": "sample"}\n', encoding="utf-8")
    args = SimpleNamespace(
        data_path=str(data_path),
        save_dir=str(tmp_path / "out"),
        batch_size=2,
        accumulation_steps=1,
        max_seq_len=128,
        dtype="bfloat16",
        use_compile=0,
        epochs=1,
        learning_rate=1e-6,
        beta=0.1,
        loss_type="sigmoid",
        tag_penalty_weight=10.0,
        from_weight="full_sft",
    )

    metadata = trainer_utils.build_checkpoint_metadata(args, _config(), "dpo")

    assert metadata["training"] == {
        "epochs": 1,
        "learning_rate": 1e-6,
        "beta": 0.1,
        "loss_type": "sigmoid",
        "tag_penalty_weight": 10.0,
        "from_weight": "full_sft",
    }


@pytest.mark.parametrize("relative_path", TRAINER_PATHS)
def test_trainers_require_explicit_resume_opt_in(relative_path):
    assert _argument_default(relative_path, "--from_resume") == 0


@pytest.mark.parametrize("relative_path", TRAINER_PATHS)
def test_trainers_expose_explicit_legacy_resume_override(relative_path):
    assert _argument_default(relative_path, "--allow_legacy_resume") == 0


@pytest.mark.parametrize("relative_path", TRAINER_PATHS)
def test_trainers_wire_metadata_and_unwrap_primary_model_on_resume(relative_path):
    calls = _checkpoint_calls(relative_path)
    load_calls = [
        call
        for call in calls
        if not any(keyword.arg == "model" for keyword in call.keywords)
    ]
    save_calls = [
        call
        for call in calls
        if any(keyword.arg == "model" for keyword in call.keywords)
    ]

    assert load_calls, f"{relative_path} must contain a checkpoint load call"
    assert save_calls, f"{relative_path} must contain a checkpoint save call"
    assert all(
        any(keyword.arg == "expected_metadata" for keyword in call.keywords)
        and any(keyword.arg == "allow_legacy_resume" for keyword in call.keywords)
        for call in load_calls
    )
    assert all(
        any(keyword.arg == "metadata" for keyword in call.keywords)
        for call in save_calls
    )
    assert _loads_primary_model_through_safe_loader(relative_path)


@pytest.mark.parametrize("relative_path", TRAINER_PATHS)
def test_trainers_explicitly_request_full_precision_stage_handoff(relative_path):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    init_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "init_model"
    ]

    assert init_calls
    assert all(
        any(keyword.arg == "resume_dir" for keyword in call.keywords)
        for call in init_calls
    )


def test_unwrap_model_recursively_removes_compile_and_ddp_wrappers(tmp_path):
    model = _TinyModel()
    wrapped = _fake_ddp(_CompiledWrapper(_CompiledWrapper(model)))

    assert trainer_utils.unwrap_model(wrapped) is model

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    critic = _CompiledWrapper(_CompiledWrapper(_TinyModel()))
    trainer_utils.lm_checkpoint(
        _config(),
        weight="wrapped",
        model=wrapped,
        optimizer=optimizer,
        save_dir=str(tmp_path),
        critic_model=critic,
    )

    inference = torch.load(
        tmp_path / "wrapped_2.pth", map_location="cpu", weights_only=True
    )
    resume = torch.load(
        tmp_path / "wrapped_2_resume.pth", map_location="cpu", weights_only=True
    )
    expected_keys = set(model.state_dict())

    assert set(inference) == expected_keys
    assert set(resume["model"]) == expected_keys
    assert set(resume["critic_model"]) == expected_keys
    assert all(
        not key.startswith(("module.", "_orig_mod."))
        for state in (inference, resume["model"], resume["critic_model"])
        for key in state
    )
    assert all(tensor.dtype == torch.float16 for tensor in inference.values())
    assert all(tensor.dtype == torch.float32 for tensor in resume["model"].values())


def test_load_model_state_normalizes_nested_legacy_wrapper_prefixes():
    source = _TinyModel()
    target = _CompiledWrapper(_CompiledWrapper(_TinyModel()))
    legacy_state = {
        f"module._orig_mod.{name}": value.detach().clone()
        for name, value in source.state_dict().items()
    }

    trainer_utils.load_model_state(target, legacy_state)

    assert torch.equal(target._orig_mod._orig_mod.linear.weight, source.linear.weight)


def test_save_inference_weights_is_atomic_and_unwraps_compiled_model(
    tmp_path, monkeypatch
):
    model = _TinyModel()
    wrapped = _CompiledWrapper(_CompiledWrapper(model))
    target = tmp_path / "inference.pth"
    real_replace = os.replace
    replacements = []

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(trainer_utils.os, "replace", record_replace)

    trainer_utils.save_inference_weights(wrapped, target)
    saved = torch.load(target, map_location="cpu", weights_only=True)

    assert set(saved) == set(model.state_dict())
    assert all(value.dtype == torch.float16 for value in saved.values())
    assert len(replacements) == 1
    assert replacements[0][0].parent == target.parent
    assert replacements[0][1] == target


@pytest.mark.parametrize("relative_path", FULL_WEIGHT_TRAINER_PATHS)
def test_trainers_delegate_inference_files_to_atomic_helper(relative_path):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    direct_torch_saves = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "save"
    ]
    atomic_saves = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "save_inference_weights"
    ]

    assert direct_torch_saves == []
    assert atomic_saves


def test_lora_adapter_save_uses_shared_atomic_writer():
    source_path = Path(__file__).resolve().parents[1] / "model/model_lora.py"
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]

    assert not any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "torch"
        and call.func.attr == "save"
        for call in calls
    )
    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "atomic_torch_save"
        for call in calls
    )


def test_checkpoint_installs_resume_before_lossy_inference_weights(
    tmp_path, monkeypatch
):
    real_replace = os.replace
    installed_targets = []

    def record_replace(source, destination):
        installed_targets.append(Path(destination).name)
        real_replace(source, destination)

    monkeypatch.setattr(trainer_utils.os, "replace", record_replace)

    _save_tiny_checkpoint(tmp_path, weight="ordered")

    assert installed_targets == [
        "ordered_2_resume.pth",
        "ordered_2.pth",
    ]


@pytest.mark.parametrize(
    ("failing_replace", "protected_name"),
    [
        (1, "atomic_2_resume.pth"),
        (2, "atomic_2.pth"),
    ],
)
def test_atomic_save_permission_error_preserves_target_and_cleans_temp(
    tmp_path, monkeypatch, failing_replace, protected_name
):
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    inference_path = save_dir / "atomic_2.pth"
    resume_path = save_dir / "atomic_2_resume.pth"
    for target in (inference_path, resume_path):
        torch.save({"sentinel": "old"}, target)

    real_replace = os.replace
    replace_calls = []

    def fail_selected_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        assert Path(source).parent == Path(destination).parent
        if len(replace_calls) == failing_replace:
            raise PermissionError("checkpoint is in use")
        real_replace(source, destination)

    monkeypatch.setattr(trainer_utils.os, "replace", fail_selected_replace)

    with pytest.raises(PermissionError, match="preserved"):
        _save_tiny_checkpoint(save_dir, weight="atomic")

    protected = torch.load(
        save_dir / protected_name, map_location="cpu", weights_only=True
    )
    assert protected == {"sentinel": "old"}
    assert len(replace_calls) == failing_replace
    assert not [path for path in save_dir.iterdir() if path.suffix == ".tmp"]


def test_resume_rejects_a_different_world_size_without_rescaling_step(
    tmp_path, monkeypatch
):
    resume_path = tmp_path / "world_2_resume.pth"
    torch.save({"world_size": 2, "step": 17}, resume_path)
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 0)

    with pytest.raises(ValueError, match="world_size"):
        trainer_utils.lm_checkpoint(
            _config(), weight="world", save_dir=str(tmp_path)
        )

    untouched = torch.load(resume_path, map_location="cpu", weights_only=True)
    assert untouched["step"] == 17


def test_resume_metadata_round_trip_and_exact_validation(tmp_path):
    metadata = {"stage": "dpo", "config_hash": "sha256:test"}
    _save_tiny_checkpoint(tmp_path, weight="metadata", metadata=metadata)

    resumed = trainer_utils.lm_checkpoint(
        _config(),
        weight="metadata",
        save_dir=str(tmp_path),
        expected_metadata=metadata,
    )

    assert resumed["metadata"] == metadata
    with pytest.raises(ValueError, match="metadata"):
        trainer_utils.lm_checkpoint(
            _config(),
            weight="metadata",
            save_dir=str(tmp_path),
            expected_metadata={"stage": "sft", "config_hash": "sha256:test"},
        )


def test_explicit_legacy_override_accepts_old_metadata_schema(tmp_path):
    _save_tiny_checkpoint(
        tmp_path,
        weight="old-schema",
        metadata={"schema_version": 1, "stage": "ppo"},
    )

    resumed = trainer_utils.lm_checkpoint(
        _config(),
        weight="old-schema",
        save_dir=str(tmp_path),
        expected_metadata={
            "schema_version": 2,
            "stage": "ppo",
            "runtime": {"sha256": "new"},
        },
        allow_legacy_resume=True,
    )

    assert resumed["metadata"]["schema_version"] == 1


def test_legacy_override_still_validates_shared_critical_fields(tmp_path):
    _save_tiny_checkpoint(
        tmp_path,
        weight="old-schema-mismatch",
        metadata={
            "schema_version": 1,
            "stage": "ppo",
            "batch_size": 8,
            "architecture": {"hidden_size": 2},
        },
    )

    with pytest.raises(ValueError, match="batch_size"):
        trainer_utils.lm_checkpoint(
            _config(),
            weight="old-schema-mismatch",
            save_dir=str(tmp_path),
            expected_metadata={
                "schema_version": 2,
                "stage": "ppo",
                "batch_size": 4,
                "architecture": {"hidden_size": 2},
            },
            allow_legacy_resume=True,
        )


def test_legacy_override_never_accepts_unknown_future_schema(tmp_path):
    _save_tiny_checkpoint(
        tmp_path,
        weight="future-schema",
        metadata={"schema_version": 99, "stage": "ppo"},
    )

    with pytest.raises(ValueError, match="future|schema"):
        trainer_utils.lm_checkpoint(
            _config(),
            weight="future-schema",
            save_dir=str(tmp_path),
            expected_metadata={"schema_version": 2, "stage": "ppo"},
            allow_legacy_resume=True,
        )


def test_resume_missing_metadata_is_rejected_unless_legacy_is_allowed(tmp_path):
    _save_tiny_checkpoint(tmp_path, weight="legacy")
    expected = {"stage": "pretrain"}

    with pytest.raises(ValueError, match="metadata"):
        trainer_utils.lm_checkpoint(
            _config(),
            weight="legacy",
            save_dir=str(tmp_path),
            expected_metadata=expected,
        )

    resumed = trainer_utils.lm_checkpoint(
        _config(),
        weight="legacy",
        save_dir=str(tmp_path),
        expected_metadata=expected,
        allow_legacy_resume=True,
    )
    assert "metadata" not in resumed


def _patch_tiny_init_model(monkeypatch):
    import model.MiniMindModel as model_module
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _path: object(),
    )
    monkeypatch.setattr(model_module, "MiniMindForCausalLM", lambda _config: _TinyModel())


def _handoff_config(**overrides):
    values = {
        "hidden_size": 2,
        "intermediate_size": 8,
        "use_moe": False,
        "dropout": 0.0,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _handoff_metadata(config, runtime_path, schema_version=2):
    return {
        "schema_version": schema_version,
        "artifact_name": "pretrain",
        "stage": "pretrain",
        "architecture": {
            field: getattr(config, field)
            for field in trainer_utils._ARCHITECTURE_METADATA_FIELDS
            if hasattr(config, field)
        },
        "runtime": trainer_utils.fingerprint_path(runtime_path),
    }


def _write_runtime_snapshot(path, content):
    path.mkdir()
    (path / "tokenizer.json").write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("field", "saved_value", "current_value"),
    (
        ("rope_theta", 10_000.0, 20_000.0),
        ("rope_scaling", None, {"factor": 2.0, "type": "linear"}),
        ("dropout", 0.0, 0.1),
    ),
)
def test_init_model_rejects_shape_compatible_architecture_drift_without_fallback(
    tmp_path, monkeypatch, field, saved_value, current_value
):
    _patch_tiny_init_model(monkeypatch)
    runtime = _write_runtime_snapshot(tmp_path / "runtime", "tokenizer-a\n")
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    saved_config = _handoff_config(**{field: saved_value})
    current_config = _handoff_config(**{field: current_value})
    torch.save(
        {
            "model": _TinyModel().state_dict(),
            "metadata": _handoff_metadata(saved_config, runtime),
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )

    with pytest.raises(ValueError, match="architecture"):
        trainer_utils.init_model(
            current_config,
            from_weight="pretrain",
            tokenizer_path=str(runtime),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_architecture_drift_rejects_inference_fallback(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    runtime = _write_runtime_snapshot(tmp_path / "runtime", "tokenizer-a\n")
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    inference_value = torch.full((2, 2), 6.5, dtype=torch.float16)
    torch.save(
        {
            "model": _TinyModel().state_dict(),
            "metadata": _handoff_metadata(
                _handoff_config(rope_theta=10_000.0), runtime
            ),
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    torch.save(
        {"linear.weight": inference_value}, output_dir / "pretrain_2.pth"
    )

    with pytest.raises(ValueError, match="architecture"):
        trainer_utils.init_model(
            _handoff_config(rope_theta=20_000.0),
            from_weight="pretrain",
            tokenizer_path=str(runtime),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_rejects_tokenizer_content_drift_without_fallback(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    saved_runtime = _write_runtime_snapshot(
        tmp_path / "runtime-a", "tokenizer-a\n"
    )
    current_runtime = _write_runtime_snapshot(
        tmp_path / "runtime-b", "tokenizer-b\n"
    )
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    config = _handoff_config()
    torch.save(
        {
            "model": _TinyModel().state_dict(),
            "metadata": _handoff_metadata(config, saved_runtime),
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )

    with pytest.raises(ValueError, match="runtime|tokenizer"):
        trainer_utils.init_model(
            config,
            from_weight="pretrain",
            tokenizer_path=str(current_runtime),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_legacy_override_validates_present_architecture_and_runtime(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    saved_runtime = _write_runtime_snapshot(
        tmp_path / "runtime-a", "tokenizer-a\n"
    )
    current_runtime = _write_runtime_snapshot(
        tmp_path / "runtime-b", "tokenizer-b\n"
    )
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    saved_config = _handoff_config(dropout=0.0)
    metadata = _handoff_metadata(saved_config, saved_runtime, schema_version=1)
    torch.save(
        {"model": _TinyModel().state_dict(), "metadata": metadata},
        checkpoint_dir / "pretrain_2_resume.pth",
    )

    with pytest.raises(ValueError, match="architecture|runtime"):
        trainer_utils.init_model(
            _handoff_config(dropout=0.1),
            from_weight="pretrain",
            tokenizer_path=str(current_runtime),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
            allow_legacy_resume=True,
        )


def test_init_model_snapshots_architecture_before_model_constructor_mutates_config(
    tmp_path, monkeypatch
):
    import model.MiniMindModel as model_module
    import transformers

    runtime = _write_runtime_snapshot(tmp_path / "runtime", "tokenizer-a\n")
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    config = _handoff_config(intermediate_size=8)
    resume_value = torch.full((2, 2), 4.25)
    torch.save(
        {
            "model": {"linear.weight": resume_value},
            "metadata": _handoff_metadata(config, runtime),
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda _path: object()
    )

    def mutating_factory(lm_config):
        lm_config.intermediate_size = 64
        return _TinyModel()

    monkeypatch.setattr(model_module, "MiniMindForCausalLM", mutating_factory)
    model, _ = trainer_utils.init_model(
        config,
        from_weight="pretrain",
        tokenizer_path=str(runtime),
        save_dir=str(output_dir),
        resume_dir=str(checkpoint_dir),
        device="cpu",
    )
    assert config.intermediate_size == 64
    assert torch.equal(model.linear.weight, resume_value)


def test_init_model_prefers_full_precision_cross_stage_resume_checkpoint(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    runtime = _write_runtime_snapshot(tmp_path / "runtime", "tokenizer-a\n")
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    config = _handoff_config()

    resume_value = torch.full((2, 2), 3.25, dtype=torch.float32)
    inference_value = torch.full((2, 2), -7.0, dtype=torch.float16)
    torch.save(
        {
            "model": {"_orig_mod.linear.weight": resume_value},
            "metadata": _handoff_metadata(config, runtime),
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    torch.save(
        {"linear.weight": inference_value}, output_dir / "pretrain_2.pth"
    )
    os.utime(output_dir / "pretrain_2.pth", ns=(1_000_000_000, 1_000_000_000))
    os.utime(
        checkpoint_dir / "pretrain_2_resume.pth",
        ns=(2_000_000_000, 2_000_000_000),
    )

    model, _ = trainer_utils.init_model(
        config,
        from_weight="pretrain",
        tokenizer_path=str(runtime),
        save_dir=str(output_dir),
        resume_dir=str(checkpoint_dir),
        device="cpu",
    )

    assert torch.equal(model.linear.weight, resume_value)


def test_init_model_rejects_unscoped_resume_even_when_inference_weight_is_newer(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(trainer_utils, "CHECKPOINT_DIR", checkpoint_dir)

    stale_resume_value = torch.full((2, 2), -4.0, dtype=torch.float32)
    latest_inference_value = torch.full((2, 2), 1.5, dtype=torch.float16)
    torch.save(
        {"model": {"linear.weight": stale_resume_value}},
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    torch.save(
        {"linear.weight": latest_inference_value},
        output_dir / "pretrain_2.pth",
    )
    os.utime(
        checkpoint_dir / "pretrain_2_resume.pth",
        ns=(1_000_000_000, 1_000_000_000),
    )
    os.utime(output_dir / "pretrain_2.pth", ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(ValueError, match="metadata|legacy"):
        trainer_utils.init_model(
            _config(),
            from_weight="pretrain",
            tokenizer_path=str(tmp_path / "tokenizer"),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_falls_back_to_inference_weights_and_strips_compiled_prefix(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(trainer_utils, "CHECKPOINT_DIR", checkpoint_dir)

    inference_value = torch.full((2, 2), 1.5, dtype=torch.float16)
    torch.save(
        {"_orig_mod.linear.weight": inference_value},
        output_dir / "pretrain_2.pth",
    )

    model, _ = trainer_utils.init_model(
        _config(),
        from_weight="pretrain",
        tokenizer_path=str(tmp_path / "tokenizer"),
        save_dir=str(output_dir),
        device="cpu",
    )

    assert torch.equal(model.linear.weight, inference_value.float())


def test_init_model_does_not_consult_global_resume_unless_requested(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "custom-out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(trainer_utils, "CHECKPOINT_DIR", checkpoint_dir)

    unrelated_resume = torch.full((2, 2), -8.0, dtype=torch.float32)
    explicit_weight = torch.full((2, 2), 2.0, dtype=torch.float16)
    torch.save(
        {"model": {"linear.weight": unrelated_resume}},
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    torch.save(
        {"linear.weight": explicit_weight},
        output_dir / "pretrain_2.pth",
    )
    os.utime(output_dir / "pretrain_2.pth", ns=(1_000_000_000, 1_000_000_000))
    os.utime(
        checkpoint_dir / "pretrain_2_resume.pth",
        ns=(2_000_000_000, 2_000_000_000),
    )

    model, _ = trainer_utils.init_model(
        _config(),
        from_weight="pretrain",
        tokenizer_path=str(tmp_path / "tokenizer"),
        save_dir=str(output_dir),
        device="cpu",
    )

    assert torch.equal(model.linear.weight, explicit_weight.float())


def test_init_model_rejects_unversioned_resume_with_inference_fallback(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "run-a"
    other_output_dir = tmp_path / "run-b"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    other_output_dir.mkdir()

    unrelated_resume = torch.full((2, 2), -9.0, dtype=torch.float32)
    explicit_weight = torch.full((2, 2), 4.0, dtype=torch.float16)
    torch.save(
        {
            "model": {"linear.weight": unrelated_resume},
            "metadata": {
                "output_dir": os.path.normcase(str(other_output_dir.resolve()))
            },
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    torch.save(
        {"linear.weight": explicit_weight}, output_dir / "pretrain_2.pth"
    )
    os.utime(output_dir / "pretrain_2.pth", ns=(1_000_000_000, 1_000_000_000))
    os.utime(
        checkpoint_dir / "pretrain_2_resume.pth",
        ns=(2_000_000_000, 2_000_000_000),
    )

    with pytest.raises(ValueError, match="schema"):
        trainer_utils.init_model(
            _config(),
            from_weight="pretrain",
            tokenizer_path=str(tmp_path / "tokenizer"),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_rejects_unscoped_resume_when_explicit_weight_exists(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "run-a"
    checkpoint_dir.mkdir()
    output_dir.mkdir()

    legacy_resume = torch.full((2, 2), -6.0, dtype=torch.float32)
    explicit_weight = torch.full((2, 2), 5.0, dtype=torch.float16)
    torch.save(
        {"model": {"linear.weight": legacy_resume}},
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    torch.save(
        {"linear.weight": explicit_weight}, output_dir / "pretrain_2.pth"
    )
    os.utime(output_dir / "pretrain_2.pth", ns=(1_000_000_000, 1_000_000_000))
    os.utime(
        checkpoint_dir / "pretrain_2_resume.pth",
        ns=(2_000_000_000, 2_000_000_000),
    )

    with pytest.raises(ValueError, match="metadata|legacy"):
        trainer_utils.init_model(
            _config(),
            from_weight="pretrain",
            tokenizer_path=str(tmp_path / "tokenizer"),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_loads_full_resume_on_cpu_before_device_transfer(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    resume_path = checkpoint_dir / "pretrain_2_resume.pth"
    resume_path.touch()
    observed_map_locations = []
    expected_state = _TinyModel().state_dict()

    def fake_load(path, map_location=None, weights_only=None):
        observed_map_locations.append((Path(path), map_location, weights_only))
        return {"model": expected_state}

    monkeypatch.setattr(trainer_utils.torch, "load", fake_load)

    trainer_utils.init_model(
        _config(),
        from_weight="pretrain",
        tokenizer_path=str(tmp_path / "tokenizer"),
        save_dir=str(output_dir),
        resume_dir=str(checkpoint_dir),
        device="meta",
        allow_legacy_resume=True,
    )

    assert observed_map_locations == [(resume_path, "cpu", True)]


def test_resolve_checkpoint_dir_scopes_default_to_output_directory(tmp_path):
    output_dir = tmp_path / "run-a"

    assert trainer_utils.resolve_checkpoint_dir(str(output_dir)) == os.path.normpath(
        str(output_dir / "checkpoints")
    )
    assert trainer_utils.resolve_checkpoint_dir(
        str(output_dir), "checkpoints"
    ) == os.path.normpath(trainer_utils.project_path("checkpoints"))


def test_resolve_lora_base_dirs_infers_parent_and_honors_explicit_overrides(
    tmp_path,
):
    adapter_dir = tmp_path / "runs" / "a" / "lora"
    inferred_base, inferred_checkpoint = trainer_utils.resolve_lora_base_dirs(
        str(adapter_dir)
    )
    assert inferred_base == os.path.normpath(str(tmp_path / "runs" / "a"))
    assert inferred_checkpoint == os.path.normpath(
        str(tmp_path / "runs" / "a" / "checkpoints")
    )

    nonstandard_adapter = tmp_path / "runs" / "adapter-a"
    same_base, same_checkpoint = trainer_utils.resolve_lora_base_dirs(
        str(nonstandard_adapter)
    )
    assert same_base == os.path.normpath(str(nonstandard_adapter))
    assert same_checkpoint == os.path.normpath(
        str(nonstandard_adapter / "checkpoints")
    )

    explicit_base = tmp_path / "base-output"
    explicit_checkpoint = tmp_path / "base-resume"
    overridden_base, overridden_checkpoint = trainer_utils.resolve_lora_base_dirs(
        str(adapter_dir), str(explicit_base), str(explicit_checkpoint)
    )
    assert overridden_base == os.path.normpath(str(explicit_base))
    assert overridden_checkpoint == os.path.normpath(str(explicit_checkpoint))


def test_lora_exposes_and_wires_explicit_base_directories():
    relative_path = "trainer/train_lora.py"
    assert _argument_default(relative_path, "--base_save_dir") is None
    assert _argument_default(relative_path, "--base_checkpoint_dir") is None
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "resolve_lora_base_dirs"
        for call in ast.walk(module)
    )


def _metadata_args(data_path, runtime_path, save_dir, **overrides):
    values = {
        "data_path": str(data_path),
        "tokenizer_path": str(runtime_path),
        "save_dir": str(save_dir),
        "save_weight": "dpo",
        "batch_size": 4,
        "accumulation_steps": 1,
        "max_seq_len": 64,
        "dtype": "bfloat16",
        "use_compile": 0,
        "epochs": 1,
        "learning_rate": 1e-6,
        "from_weight": "full_sft",
        "logprob_reduction": "mean",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_metadata_identity_ignores_mount_paths_and_mtimes_but_not_content(
    tmp_path,
):
    first_root = tmp_path / "node-a"
    second_root = tmp_path / "node-b"
    for root in (first_root, second_root):
        (root / "runtime").mkdir(parents=True)
        (root / "train.jsonl").write_text("same data\n", encoding="utf-8")
        (root / "runtime" / "tokenizer.json").write_text(
            "same tokenizer\n", encoding="utf-8"
        )
    os.utime(second_root / "train.jsonl", ns=(4_000_000_000, 4_000_000_000))

    first = trainer_utils.build_checkpoint_metadata(
        _metadata_args(
            first_root / "train.jsonl", first_root / "runtime", first_root / "out"
        ),
        _config(),
        "dpo",
    )
    second = trainer_utils.build_checkpoint_metadata(
        _metadata_args(
            second_root / "train.jsonl",
            second_root / "runtime",
            second_root / "different-out",
        ),
        _config(),
        "dpo",
    )

    _save_tiny_checkpoint(tmp_path / "checkpoint", weight="portable", metadata=first)
    resumed = trainer_utils.lm_checkpoint(
        _config(),
        weight="portable",
        save_dir=str(tmp_path / "checkpoint"),
        expected_metadata=second,
    )
    assert resumed["metadata"] == first

    (second_root / "train.jsonl").write_text("changed data\n", encoding="utf-8")
    changed = trainer_utils.build_checkpoint_metadata(
        _metadata_args(
            second_root / "train.jsonl",
            second_root / "runtime",
            second_root / "different-out",
        ),
        _config(),
        "dpo",
    )
    with pytest.raises(ValueError, match="metadata"):
        trainer_utils.lm_checkpoint(
            _config(),
            weight="portable",
            save_dir=str(tmp_path / "checkpoint"),
            expected_metadata=changed,
        )


def test_checkpoint_metadata_normalizes_lora_targets_and_dpo_reduction(tmp_path):
    data_path = tmp_path / "train.jsonl"
    runtime_path = tmp_path / "runtime"
    data_path.write_text("data\n", encoding="utf-8")
    runtime_path.mkdir()
    (runtime_path / "tokenizer.json").write_text("tokenizer\n", encoding="utf-8")

    dpo_metadata = trainer_utils.build_checkpoint_metadata(
        _metadata_args(data_path, runtime_path, tmp_path / "out"),
        _config(),
        "dpo",
    )
    lora_metadata = trainer_utils.build_checkpoint_metadata(
        _metadata_args(
            data_path,
            runtime_path,
            tmp_path / "out-lora",
            save_weight=None,
            lora_name="adapter-a",
            lora_rank=8,
            lora_alpha=16,
            lora_target_modules=["v_proj", "q_proj", "k_proj"],
        ),
        _config(),
        "lora",
    )

    assert dpo_metadata["artifact_name"] == "dpo"
    assert dpo_metadata["training"]["logprob_reduction"] == "mean"
    assert lora_metadata["artifact_name"] == "adapter-a"
    assert lora_metadata["architecture"]["adapter"]["target_modules"] == [
        "k_proj",
        "q_proj",
        "v_proj",
    ]


def test_distributed_fingerprint_elects_one_leader_per_host_without_local_rank(
    tmp_path, monkeypatch
):
    dependency = tmp_path / "data.txt"
    dependency.write_text("shared\n", encoding="utf-8")
    expected = trainer_utils.fingerprint_path(str(dependency))
    calls = []

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(trainer_utils.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(trainer_utils.socket, "gethostname", lambda: "node-b")

    def fake_all_gather(gathered, value):
        calls.append(value)
        if len(calls) == 1:
            gathered[:] = ["node-a", "node-a", "node-b", "node-b"]
        else:
            gathered[:] = [
                {"fingerprint": expected},
                None,
                value,
                None,
            ]

    monkeypatch.setattr(trainer_utils.dist, "all_gather_object", fake_all_gather)
    assert trainer_utils.distributed_fingerprint_path(str(dependency)) == expected
    assert calls[1] == {"fingerprint": expected}


def test_coordinated_checkpoint_save_preserves_single_process_exception(monkeypatch):
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: False)
    derived_calls = []

    class PrimaryFailure(Exception):
        pass

    def fail_primary():
        raise PrimaryFailure("resume failed")

    with pytest.raises(PrimaryFailure, match="resume failed"):
        trainer_utils.coordinated_checkpoint_save(
            fail_primary, lambda: derived_calls.append("derived")
        )
    assert derived_calls == []


def test_coordinated_checkpoint_save_broadcasts_rank_zero_failure(monkeypatch):
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 0)
    broadcasts = []
    derived_calls = []

    def broadcast(status, src):
        broadcasts.append((dict(status[0]), src))

    monkeypatch.setattr(trainer_utils.dist, "broadcast_object_list", broadcast)

    with pytest.raises(RuntimeError, match="OSError: disk full"):
        trainer_utils.coordinated_checkpoint_save(
            lambda: (_ for _ in ()).throw(OSError("disk full")),
            lambda: derived_calls.append("derived"),
        )
    assert broadcasts == [
        ({"ok": False, "error": "OSError: disk full"}, 0)
    ]
    assert derived_calls == []


@pytest.mark.parametrize(
    ("primary_save", "derived_save", "error_type", "message"),
    (
        (
            lambda: (_ for _ in ()).throw(KeyboardInterrupt("stop primary")),
            lambda: pytest.fail("derived must not run after primary failure"),
            KeyboardInterrupt,
            "stop primary",
        ),
        (
            lambda: None,
            lambda: (_ for _ in ()).throw(SystemExit("stop derived")),
            SystemExit,
            "stop derived",
        ),
    ),
)
def test_coordinated_checkpoint_save_broadcasts_base_exception_before_rank_zero_reraises(
    monkeypatch, primary_save, derived_save, error_type, message
):
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 0)
    broadcasts = []
    monkeypatch.setattr(
        trainer_utils.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((dict(status[0]), src)),
    )

    with pytest.raises(error_type, match=message):
        trainer_utils.coordinated_checkpoint_save(primary_save, derived_save)

    assert broadcasts == [
        (
            {"ok": False, "error": f"{error_type.__name__}: {message}"},
            0,
        )
    ]


def test_coordinated_checkpoint_save_nonzero_rank_raises_for_broadcast_base_exception(
    monkeypatch,
):
    monkeypatch.setattr(trainer_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_utils.dist, "get_rank", lambda: 1)

    def broadcast(status, src):
        assert status == [None]
        assert src == 0
        status[0] = {"ok": False, "error": "KeyboardInterrupt: rank zero stopped"}

    monkeypatch.setattr(trainer_utils.dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="KeyboardInterrupt: rank zero stopped"):
        trainer_utils.coordinated_checkpoint_save(
            lambda: pytest.fail("nonzero rank must not save"),
            lambda: pytest.fail("nonzero rank must not save derived artifact"),
        )


def test_reference_sidecar_reuse_repairs_changed_or_corrupt_file(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    reference = _TinyModel()
    _save_tiny_checkpoint(
        checkpoint_dir, weight="repair-sidecar", ref_model=reference
    )
    resume_path = checkpoint_dir / "repair-sidecar_2_resume.pth"
    descriptor = torch.load(
        resume_path, map_location="cpu", weights_only=True
    )["ref_model"]
    sidecar_path = checkpoint_dir / descriptor["file"]

    def corrupt_sidecar():
        prior_mtime = sidecar_path.stat().st_mtime_ns
        torch.save({"wrong": torch.ones(7)}, sidecar_path)
        os.utime(sidecar_path, ns=(prior_mtime + 1_000_000_000,) * 2)

    corrupt_sidecar()
    trainer_utils._REFERENCE_SIDECAR_CACHE.clear()
    _save_tiny_checkpoint(
        checkpoint_dir, weight="repair-sidecar", ref_model=reference
    )
    repaired = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    assert trainer_utils._state_dict_sha256(repaired) == descriptor["state_sha256"]

    corrupt_sidecar()
    _save_tiny_checkpoint(
        checkpoint_dir, weight="repair-sidecar", ref_model=reference
    )
    repaired_again = torch.load(
        sidecar_path, map_location="cpu", weights_only=True
    )
    assert (
        trainer_utils._state_dict_sha256(repaired_again)
        == descriptor["state_sha256"]
    )


def test_init_model_rejects_mismatched_resume_without_inference_fallback(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    torch.save(
        {
            "model": _TinyModel().state_dict(),
            "metadata": {
                "schema_version": 2,
                "artifact_name": "full_sft",
                "stage": "full_sft",
            },
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )

    with pytest.raises((ValueError, FileNotFoundError), match="artifact|pretrain|source"):
        trainer_utils.init_model(
            _config(),
            from_weight="pretrain",
            tokenizer_path=str(tmp_path / "tokenizer"),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_rejects_inference_fallback_for_mismatched_resume(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    inference_value = torch.full((2, 2), 2.5, dtype=torch.float16)
    torch.save(
        {
            "model": _TinyModel().state_dict(),
            "metadata": {
                "schema_version": 2,
                "artifact_name": "full_sft",
                "stage": "full_sft",
            },
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )
    torch.save(
        {"linear.weight": inference_value}, output_dir / "pretrain_2.pth"
    )

    with pytest.raises(ValueError, match="artifact"):
        trainer_utils.init_model(
            _config(),
            from_weight="pretrain",
            tokenizer_path=str(tmp_path / "tokenizer"),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
        )


def test_init_model_legacy_override_checks_present_artifact_identity(
    tmp_path, monkeypatch
):
    _patch_tiny_init_model(monkeypatch)
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "out"
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    torch.save(
        {
            "model": _TinyModel().state_dict(),
            "metadata": {
                "schema_version": 1,
                "artifact_name": "full_sft",
                "stage": "full_sft",
            },
        },
        checkpoint_dir / "pretrain_2_resume.pth",
    )

    with pytest.raises((ValueError, FileNotFoundError), match="artifact|pretrain"):
        trainer_utils.init_model(
            _config(),
            from_weight="pretrain",
            tokenizer_path=str(tmp_path / "tokenizer"),
            save_dir=str(output_dir),
            resume_dir=str(checkpoint_dir),
            device="cpu",
            allow_legacy_resume=True,
        )


def _parent_map(module):
    parents = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


@pytest.mark.parametrize("relative_path", TRAINER_PATHS)
def test_trainers_scope_checkpoint_directory_and_coordinate_saves(relative_path):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    parents = _parent_map(module)

    assert _argument_default(relative_path, "--checkpoint_dir") is None
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "resolve_checkpoint_dir"
        for call in ast.walk(module)
    ), f"{relative_path} must resolve its stage-local checkpoint directory"

    checkpoint_calls = _checkpoint_calls(relative_path)
    assert checkpoint_calls
    for call in checkpoint_calls:
        save_dir = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "save_dir"),
            None,
        )
        assert save_dir is not None
        assert not (
            isinstance(save_dir, ast.Constant) and save_dir.value == "checkpoints"
        ), f"{relative_path} still hardcodes the shared checkpoint namespace"

    coordinated = [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "coordinated_checkpoint_save"
    ]
    assert coordinated, f"{relative_path} must coordinate periodic/final saves"
    for call in coordinated:
        callbacks = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(callbacks.get("primary_save"), ast.Lambda)
        assert isinstance(callbacks.get("derived_save"), ast.Lambda)
        assert any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "lm_checkpoint"
            for inner in ast.walk(callbacks["primary_save"])
        ), f"{relative_path} must install the full-precision resume first"
        derived_helper = (
            "save_lora" if relative_path == "trainer/train_lora.py"
            else "save_inference_weights"
        )
        assert any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == derived_helper
            for inner in ast.walk(callbacks["derived_save"])
        ), f"{relative_path} must save its derived artifact second"
        ancestor = parents.get(call)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                assert not any(
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "is_main_process"
                    for inner in ast.walk(ancestor.test)
                ), f"{relative_path} keeps coordinated save inside a rank-zero guard"
            ancestor = parents.get(ancestor)

    init_calls = [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "init_model"
    ]
    assert init_calls
    assert all(
        any(keyword.arg == "allow_legacy_resume" for keyword in call.keywords)
        for call in init_calls
    ), f"{relative_path} must pass the explicit legacy policy to init_model"


def test_lora_resume_is_strict_and_uses_separate_base_checkpoint_scope():
    source_path = Path(__file__).resolve().parents[1] / "trainer/train_lora.py"
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    strict_loads = [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "load_model_state"
        and any(
            keyword.arg == "strict"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    ]
    assert strict_loads
    adapter_scope_calls = [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "resolve_checkpoint_dir"
    ]
    base_scope_calls = [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "resolve_lora_base_dirs"
    ]
    assert adapter_scope_calls and base_scope_calls, (
        "LoRA must resolve its adapter checkpoint scope separately from the "
        "base full_sft handoff scope"
    )


@pytest.mark.parametrize(
    "relative_path",
    TRAINER_PATHS,
)
def test_trainers_have_independent_final_coordinated_save(relative_path):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    parents = _parent_map(module)
    coordinated = [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "coordinated_checkpoint_save"
    ]
    final_saves = []
    for call in coordinated:
        ancestor = parents.get(call)
        inside_loop = False
        while ancestor is not None:
            inside_loop = inside_loop or isinstance(ancestor, (ast.For, ast.While))
            ancestor = parents.get(ancestor)
        if not inside_loop:
            final_saves.append(call)
    assert final_saves, (
        f"{relative_path} needs a final coordinated save outside the training "
        "loop so a failed derived artifact can be repaired after exact resume"
    )
