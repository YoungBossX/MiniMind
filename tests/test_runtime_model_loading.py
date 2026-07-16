import pytest
import ast
from pathlib import Path


torch = pytest.importorskip("torch")
from torch import nn

from model.MiniMindModel import MiniMindConfig
from evals.core.load_model import load_model_and_tokenizer
from trainer.train_ppo import CriticModel


class _BackboneWithFinalOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, **_kwargs):
        hidden_states = torch.ones(
            input_ids.shape[0], input_ids.shape[1], self.hidden_size
        )
        return hidden_states, None, torch.tensor(0.0)

    def norm(self, _hidden_states):
        raise AssertionError("Critic must not normalize the backbone final output twice")


def test_ppo_critic_consumes_backbone_final_hidden_states_directly():
    config = MiniMindConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
    )
    critic = CriticModel(config)
    critic.model = _BackboneWithFinalOutput(config.hidden_size)

    values = critic(input_ids=torch.tensor([[3, 4, 5]]))

    assert values.shape == (1, 3)


def test_explicit_missing_lora_path_raises_file_not_found(tmp_path):
    missing_lora = tmp_path / "missing-adapter.pth"

    with pytest.raises(FileNotFoundError, match="LoRA checkpoint not found"):
        load_model_and_tokenizer(
            hidden_size=16,
            num_hidden_layers=1,
            lora_path=str(missing_lora),
        )


@pytest.mark.parametrize("relative_path", [
    "evals/core/load_model.py",
    "model/model_lora.py",
    "eval.py",
    "eval/benchmark.py",
    "trainer/trainer_utils.py",
    "trainer/train_ppo.py",
])
def test_inference_weight_loaders_restrict_unpickling_to_weights(relative_path):
    source = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(source.read_text(encoding="utf-8-sig"))
    torch_load_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]

    assert torch_load_calls
    assert all(
        any(
            keyword.arg == "weights_only"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in torch_load_calls
    )


def test_external_interactive_model_code_requires_explicit_opt_in():
    source = Path(__file__).resolve().parents[1] / "eval.py"
    module = ast.parse(source.read_text(encoding="utf-8-sig"))
    external_model_loads = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name)
        and node.args[0].value.id == "args"
        and node.args[0].attr == "load_from"
    ]

    assert len(external_model_loads) == 1
    assert any(
        keyword.arg == "trust_remote_code"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "args"
        and keyword.value.attr == "trust_remote_code"
        for keyword in external_model_loads[0].keywords
    )
