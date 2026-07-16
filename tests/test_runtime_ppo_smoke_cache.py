import importlib
import shutil
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")
dynamic_module_utils = pytest.importorskip("transformers.dynamic_module_utils")
transformers = pytest.importorskip("transformers")


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
TEMP_CACHE_ROOT = EVAL_DIR / ".tmp"


def assert_project_local_cache(cache_dir):
    cache_dir = Path(cache_dir).resolve()
    assert cache_dir.is_relative_to(TEMP_CACHE_ROOT.resolve())
    assert cache_dir.name.startswith("huggingface_modules_")


@pytest.fixture(scope="module")
def smoke_test_module():
    sys.path.insert(0, str(EVAL_DIR))
    try:
        return importlib.import_module("smoke_test")
    finally:
        sys.path.remove(str(EVAL_DIR))


def test_ppo_smoke_cache_redirects_dynamic_modules_and_cleans_up_on_success(smoke_test_module):
    original_cache = dynamic_module_utils.HF_MODULES_CACHE

    with smoke_test_module.temporary_hf_modules_cache() as configured_cache:
        configured_cache = Path(configured_cache).resolve()
        assert_project_local_cache(configured_cache)
        assert Path(dynamic_module_utils.HF_MODULES_CACHE).resolve() == configured_cache
        (configured_cache / "sentinel.py").write_text("# temporary module cache\n", encoding="utf-8")

    assert dynamic_module_utils.HF_MODULES_CACHE == original_cache
    assert not configured_cache.exists()


def test_ppo_smoke_cache_restores_and_cleans_up_when_loading_raises(smoke_test_module):
    original_cache = dynamic_module_utils.HF_MODULES_CACHE

    with pytest.raises(RuntimeError, match="reward model load failed"):
        with smoke_test_module.temporary_hf_modules_cache() as configured_cache:
            configured_cache = Path(configured_cache).resolve()
            assert_project_local_cache(configured_cache)
            assert Path(dynamic_module_utils.HF_MODULES_CACHE).resolve() == Path(configured_cache).resolve()
            (Path(configured_cache) / "partial.py").write_text("# partial module\n", encoding="utf-8")
            raise RuntimeError("reward model load failed")

    assert dynamic_module_utils.HF_MODULES_CACHE == original_cache
    assert not configured_cache.exists()


def test_ppo_reward_component_loader_redirects_every_remote_code_load(smoke_test_module, monkeypatch):
    original_cache = dynamic_module_utils.HF_MODULES_CACHE
    observed_caches = []

    class FakeRewardModel:
        def to(self, device):
            assert device == "cuda:0"
            return self

        def eval(self):
            return self

    fake_reward_model = FakeRewardModel()
    fake_reward_tokenizer = object()

    def fake_model_loader(*_args, **kwargs):
        assert kwargs["trust_remote_code"] is True
        observed_caches.append(Path(dynamic_module_utils.HF_MODULES_CACHE).resolve())
        return fake_reward_model

    def fake_tokenizer_loader(*_args, **kwargs):
        assert kwargs["trust_remote_code"] is True
        observed_caches.append(Path(dynamic_module_utils.HF_MODULES_CACHE).resolve())
        return fake_reward_tokenizer

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tokenizer_loader)

    reward_model, reward_tokenizer = smoke_test_module.load_ppo_reward_components(
        "local-reward-model", "cuda:0"
    )

    assert reward_model is fake_reward_model
    assert reward_tokenizer is fake_reward_tokenizer
    assert observed_caches[0] == observed_caches[1]
    assert_project_local_cache(observed_caches[0])
    assert dynamic_module_utils.HF_MODULES_CACHE == original_cache
    assert not observed_caches[0].exists()


def test_ppo_smoke_cache_removes_sys_path_entry_created_by_transformers(smoke_test_module):
    original_cache = dynamic_module_utils.HF_MODULES_CACHE

    with smoke_test_module.temporary_hf_modules_cache() as configured_cache:
        configured_cache = Path(configured_cache).resolve()
        dynamic_module_utils.init_hf_modules()
        assert str(configured_cache) in sys.path

    try:
        assert dynamic_module_utils.HF_MODULES_CACHE == original_cache
        assert str(configured_cache) not in sys.path
        assert not configured_cache.exists()
    finally:
        while str(configured_cache) in sys.path:
            sys.path.remove(str(configured_cache))


def test_ppo_smoke_cache_nested_contexts_are_isolated_and_restore_outer_cache(smoke_test_module):
    original_cache = dynamic_module_utils.HF_MODULES_CACHE

    with smoke_test_module.temporary_hf_modules_cache() as outer_cache:
        outer_cache = Path(outer_cache).resolve()
        with smoke_test_module.temporary_hf_modules_cache() as inner_cache:
            inner_cache = Path(inner_cache).resolve()
            assert outer_cache != inner_cache
            assert_project_local_cache(outer_cache)
            assert_project_local_cache(inner_cache)
            assert Path(dynamic_module_utils.HF_MODULES_CACHE).resolve() == inner_cache

        assert Path(dynamic_module_utils.HF_MODULES_CACHE).resolve() == outer_cache
        assert outer_cache.exists()
        assert not inner_cache.exists()

    assert dynamic_module_utils.HF_MODULES_CACHE == original_cache
    assert not outer_cache.exists()


def test_ppo_smoke_cache_preserves_body_error_when_cleanup_fails(smoke_test_module, monkeypatch, capsys):
    original_cache = dynamic_module_utils.HF_MODULES_CACHE
    real_rmtree = smoke_test_module.shutil.rmtree
    cache_dir = None

    def fail_cleanup(_path):
        raise OSError("forced cache cleanup failure")

    monkeypatch.setattr(smoke_test_module.shutil, "rmtree", fail_cleanup)
    try:
        with pytest.warns(RuntimeWarning, match="Failed to clean up PPO Hugging Face module cache"):
            with pytest.raises(ValueError, match="reward model load failed"):
                with smoke_test_module.temporary_hf_modules_cache() as configured_cache:
                    cache_dir = Path(configured_cache)
                    raise ValueError("reward model load failed")
    finally:
        if cache_dir is not None:
            real_rmtree(cache_dir, ignore_errors=True)

    assert dynamic_module_utils.HF_MODULES_CACHE == original_cache
    assert "Failed to clean up PPO Hugging Face module cache" in capsys.readouterr().err
