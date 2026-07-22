import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from model.MiniMindModel import (
    MiniMindConfig,
    MiniMindForCausalLM,
    MoEFeedForward,
    MoEGate,
)


def _config(**overrides):
    values = {
        "dropout": 0.0,
        "hidden_size": 16,
        "intermediate_size": 32,
        "max_position_embeddings": 32,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32,
    }
    values.update(overrides)
    return MiniMindConfig(**values)


def _moe_config(seq_aux):
    return _config(
        use_moe=True,
        num_hidden_layers=1,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        aux_loss_alpha=0.1,
        seq_aux=seq_aux,
    )


@pytest.mark.parametrize(
    "mask_values",
    [
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]],
    ],
    ids=["right-padding", "left-padding"],
)
def test_masked_prefill_combines_causal_and_padding_masks_for_sdpa(
    monkeypatch, mask_values
):
    torch.manual_seed(0)
    model = MiniMindForCausalLM(_config(flash_attention=True)).eval()
    input_ids = torch.tensor(
        [[5, 6, 7, 0, 0], [8, 9, 10, 11, 0]], dtype=torch.long
    )
    attention_mask = torch.tensor(mask_values, dtype=torch.long)
    observed_sdpa = []
    observed_all = []
    original_sdpa = F.scaled_dot_product_attention
    original_all = torch.all

    def recording_sdpa(query, key, value, **kwargs):
        observed_sdpa.append(
            {
                "mask": kwargs.get("attn_mask").detach().clone(),
                "is_causal": kwargs.get("is_causal"),
            }
        )
        return original_sdpa(query, key, value, **kwargs)

    def recording_all(*args, **kwargs):
        observed_all.append(True)
        return original_all(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    monkeypatch.setattr(torch, "all", recording_all)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    sequence_length = attention_mask.size(1)
    causal_mask = torch.ones(
        sequence_length, sequence_length, dtype=torch.bool
    ).tril()
    expected_mask = (
        attention_mask[:, None, None, :].bool()
        & causal_mask[None, None, :, :]
    )
    assert len(observed_sdpa) == model.config.num_hidden_layers
    assert observed_all == []
    for call in observed_sdpa:
        assert call["is_causal"] is False
        assert call["mask"].dtype == torch.bool
        assert torch.equal(call["mask"], expected_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_masked_sdpa_runs_with_cuda_math_backend():
    torch.manual_seed(5)
    model = MiniMindForCausalLM(_config(flash_attention=True)).cuda().eval()
    input_ids = torch.tensor(
        [[5, 6, 7, 0], [8, 9, 10, 11]], device="cuda", dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 1, 1]], device="cuda", dtype=torch.long
    )

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

    assert torch.isfinite(logits[attention_mask.bool()]).all()


@pytest.mark.parametrize(
    ("input_ids", "attention_mask"),
    [
        (
            [[5, 6, 7, 0, 0], [8, 9, 10, 11, 0]],
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        ),
        (
            [[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]],
            [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]],
        ),
    ],
    ids=["right-padding", "left-padding"],
)
def test_masked_sdpa_matches_manual_attention_on_valid_tokens(input_ids, attention_mask):
    torch.manual_seed(1)
    flash_model = MiniMindForCausalLM(_config(flash_attention=True)).eval()
    manual_model = MiniMindForCausalLM(_config(flash_attention=False)).eval()
    manual_model.load_state_dict(flash_model.state_dict())
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    with torch.no_grad():
        flash_logits = flash_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        manual_logits = manual_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

    valid_tokens = attention_mask.bool()
    torch.testing.assert_close(
        flash_logits[valid_tokens],
        manual_logits[valid_tokens],
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("seq_aux", [True, False], ids=["sequence", "token"])
def test_moe_gate_excludes_padding_from_aux_loss_without_dropping_routes(seq_aux):
    torch.manual_seed(2)
    gate = MoEGate(_moe_config(seq_aux)).train()
    valid_hidden = torch.randn(2, 3, gate.gating_dim)
    padding_hidden = torch.full((2, 2, gate.gating_dim), 7.0)
    padded_hidden = torch.cat([valid_hidden, padding_hidden], dim=1)
    valid_mask = torch.ones(2, 3, dtype=torch.long)
    padded_mask = torch.cat([valid_mask, torch.zeros(2, 2, dtype=torch.long)], dim=1)

    base_idx, base_weight, base_aux = gate(valid_hidden, attention_mask=valid_mask)
    padded_idx, padded_weight, padded_aux = gate(
        padded_hidden, attention_mask=padded_mask
    )

    top_k = gate.top_k
    assert base_idx.shape == (2 * 3, top_k)
    assert base_weight.shape == (2 * 3, top_k)
    assert padded_idx.shape == (2 * 5, top_k)
    assert padded_weight.shape == (2 * 5, top_k)
    assert torch.equal(
        padded_idx.view(2, 5, top_k)[:, :3], base_idx.view(2, 3, top_k)
    )
    torch.testing.assert_close(
        padded_weight.view(2, 5, top_k)[:, :3],
        base_weight.view(2, 3, top_k),
    )
    torch.testing.assert_close(padded_aux, base_aux)


def test_moe_seq_aux_handles_zero_count_without_clamping_expected_load():
    gate = MoEGate(_moe_config(seq_aux=True)).train()
    hidden_states = torch.zeros(2, 2, gate.gating_dim)
    attention_mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.long)

    _, _, aux_loss = gate(hidden_states, attention_mask=attention_mask)

    assert torch.isfinite(aux_loss)
    torch.testing.assert_close(aux_loss, aux_loss.new_tensor(gate.alpha))


@pytest.mark.parametrize("seq_aux", [True, False], ids=["sequence", "token"])
def test_moe_feed_forward_ignores_padding_for_aux_loss_and_keeps_output_shape(seq_aux):
    torch.manual_seed(3)
    moe = MoEFeedForward(_moe_config(seq_aux)).train()
    valid_hidden = torch.randn(2, 3, moe.config.hidden_size)
    padding_hidden = torch.full((2, 2, moe.config.hidden_size), -9.0)
    padded_hidden = torch.cat([valid_hidden, padding_hidden], dim=1)
    valid_mask = torch.ones(2, 3, dtype=torch.long)
    padded_mask = torch.cat([valid_mask, torch.zeros(2, 2, dtype=torch.long)], dim=1)

    base_output = moe(valid_hidden, attention_mask=valid_mask)
    base_aux = moe.aux_loss.detach().clone()
    padded_output = moe(padded_hidden, attention_mask=padded_mask)
    padded_aux = moe.aux_loss.detach().clone()

    assert base_output.shape == valid_hidden.shape
    assert padded_output.shape == padded_hidden.shape
    torch.testing.assert_close(padded_output[:, :3], base_output)
    torch.testing.assert_close(padded_aux, base_aux)


def test_moe_block_passes_only_the_current_query_mask_during_cached_generation(
    monkeypatch,
):
    torch.manual_seed(4)
    model = MiniMindForCausalLM(_moe_config(seq_aux=True)).eval()
    mlp = model.model.block_layers[0].mlp
    original_forward = mlp.forward
    observed_masks = []

    def recording_forward(hidden_states, attention_mask=None):
        observed_masks.append(
            None if attention_mask is None else attention_mask.detach().clone()
        )
        if attention_mask is None:
            return original_forward(hidden_states)
        return original_forward(hidden_states, attention_mask=attention_mask)

    monkeypatch.setattr(mlp, "forward", recording_forward)
    prompt_ids = torch.tensor([[0, 5, 6]], dtype=torch.long)
    prompt_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)

    with torch.no_grad():
        prefill = model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=True,
        )
        model(
            input_ids=torch.tensor([[7]], dtype=torch.long),
            attention_mask=torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )

    assert len(observed_masks) == 2
    assert torch.equal(observed_masks[0], prompt_mask)
    assert torch.equal(observed_masks[1], torch.ones((1, 1), dtype=torch.long))


def test_cached_single_token_generation_stays_manual_and_matches_full_forward(
    monkeypatch,
):
    torch.manual_seed(5)
    model = MiniMindForCausalLM(_config(flash_attention=True)).eval()
    prompt_ids = torch.tensor([[0, 5, 6], [8, 9, 10]], dtype=torch.long)
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long)
    next_ids = torch.tensor([[7], [11]], dtype=torch.long)
    full_ids = torch.cat([prompt_ids, next_ids], dim=1)
    full_mask = torch.cat(
        [prompt_mask, torch.ones((2, 1), dtype=torch.long)], dim=1
    )
    observed_sdpa = []
    original_sdpa = F.scaled_dot_product_attention

    def recording_sdpa(query, key, value, **kwargs):
        observed_sdpa.append(True)
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)

    with torch.no_grad():
        prefill = model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=True,
        )
        observed_sdpa.clear()
        cached = model(
            input_ids=next_ids,
            attention_mask=full_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        assert observed_sdpa == []
        full = model(input_ids=full_ids, attention_mask=full_mask)

    torch.testing.assert_close(
        cached.logits[:, -1],
        full.logits[:, -1],
        rtol=1e-4,
        atol=1e-5,
    )
