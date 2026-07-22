import ast
import inspect
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from trainer import train_grpo, train_ppo, trainer_utils
from dataset import llm_dataset


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def local_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(ROOT / "model")


def _parser_default(relative_path, option):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == option:
            defaults = [kw.value for kw in node.keywords if kw.arg == "default"]
            assert len(defaults) == 1, f"{option} must have exactly one default"
            return ast.literal_eval(defaults[0])
    pytest.fail(f"{relative_path} does not define {option}")


def test_compute_gae_resets_at_terminal_and_does_not_bootstrap_padding():
    rewards = torch.tensor([5.0])
    values = torch.tensor([[1.0, 2.0, 100.0, 200.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    advantages, returns = train_ppo.compute_gae(
        rewards, values, response_mask, gamma=1.0, lam=0.5
    )

    torch.testing.assert_close(advantages, torch.tensor([[2.5, 3.0, 0.0, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[3.5, 5.0, 0.0, 0.0]]))


def test_compute_gae_isolated_from_nonfinite_padding_values():
    advantages, returns = train_ppo.compute_gae(
        torch.tensor([5.0]),
        torch.tensor([[1.0, 2.0, float("nan"), float("inf")]]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        gamma=1.0,
        lam=0.5,
    )

    torch.testing.assert_close(advantages, torch.tensor([[2.5, 3.0, 0.0, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[3.5, 5.0, 0.0, 0.0]]))


def test_compute_gae_handles_a_fully_padded_response():
    advantages, returns = train_ppo.compute_gae(
        torch.tensor([7.0]),
        torch.tensor([[2.0, 3.0]]),
        torch.zeros(1, 2),
    )

    assert torch.count_nonzero(advantages) == 0
    assert torch.count_nonzero(returns) == 0


def test_compute_gae_preserves_value_dtype():
    values = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    advantages, returns = train_ppo.compute_gae(
        torch.tensor([3.0], dtype=torch.float64), values, torch.ones_like(values)
    )

    assert advantages.dtype == values.dtype
    assert returns.dtype == values.dtype


def test_clamp_log_ratio_bounds_forward_but_preserves_extreme_gradient():
    log_ratio = torch.tensor([-1_000.0, 1_000.0], requires_grad=True)

    bounded = trainer_utils.clamp_log_ratio(log_ratio)
    (bounded * torch.tensor([2.0, -3.0])).sum().backward()

    torch.testing.assert_close(bounded, torch.tensor([-20.0, 20.0]))
    torch.testing.assert_close(log_ratio.grad, torch.tensor([2.0, -3.0]))


def test_clamp_log_ratio_keeps_infinite_inputs_bounded_with_finite_gradients():
    log_ratio = torch.tensor([-float("inf"), float("inf")], requires_grad=True)

    bounded = trainer_utils.clamp_log_ratio(log_ratio)
    bounded.sum().backward()

    torch.testing.assert_close(bounded, torch.tensor([-20.0, 20.0]))
    torch.testing.assert_close(log_ratio.grad, torch.ones(2))


def test_rl_prompt_tokenization_keeps_newest_complete_chatml_turns(
    local_tokenizer,
):
    messages = [
        "<|im_start|>system\nold system context<|im_end|>\n",
        "<|im_start|>user\nold question<|im_end|>\n",
        "<|im_start|>assistant\nrecent answer<|im_end|>\n",
        "<|im_start|>user\nnewest question<|im_end|>\n",
    ]
    generation_marker = "<|im_start|>assistant\n"
    prompt = "".join(messages) + generation_marker
    expected_prompt = "".join(messages[-2:]) + generation_marker
    max_length = len(
        local_tokenizer(expected_prompt, add_special_tokens=False).input_ids
    )

    encoded, actor_prompts = trainer_utils.tokenize_rl_prompts(
        local_tokenizer, [prompt], max_length=max_length, device="cpu"
    )

    assert actor_prompts == [expected_prompt]
    expected_ids = local_tokenizer(
        expected_prompt, add_special_tokens=False
    ).input_ids
    actual_ids = encoded.input_ids[0][encoded.attention_mask[0].bool()].tolist()
    assert actual_ids == expected_ids


def test_rl_prompt_tokenization_preserves_long_message_chatml_boundaries(
    local_tokenizer,
):
    header = "<|im_start|>user\n"
    end_marker = "<|im_end|>\n"
    generation_marker = "<|im_start|>assistant\n"
    content_tail = "critical newest content tail"
    prompt = (
        header
        + ("discarded old content " * 200)
        + content_tail
        + end_marker
        + generation_marker
    )
    max_length = len(
        local_tokenizer(
            header + content_tail + end_marker + generation_marker,
            add_special_tokens=False,
        ).input_ids
    )

    encoded, actor_prompts = trainer_utils.tokenize_rl_prompts(
        local_tokenizer, [prompt], max_length=max_length, device="cpu"
    )

    retained = actor_prompts[0]
    assert retained.startswith(header)
    assert retained.endswith(content_tail + end_marker + generation_marker)
    assert "discarded old content discarded" not in retained
    assert encoded.attention_mask.sum().item() <= max_length


def test_rl_prompt_tokenization_allows_exact_boundary_only_budget(
    local_tokenizer,
):
    header = "<|im_start|>user\n"
    end_marker = "<|im_end|>\n"
    generation_marker = "<|im_start|>assistant\n"
    boundary_only_prompt = header + end_marker + generation_marker
    prompt = (
        header
        + ("overlong content " * 100)
        + "sentinel"
        + end_marker
        + generation_marker
    )
    max_length = len(
        local_tokenizer(
            boundary_only_prompt, add_special_tokens=False
        ).input_ids
    )

    encoded, actor_prompts = trainer_utils.tokenize_rl_prompts(
        local_tokenizer, [prompt], max_length=max_length, device="cpu"
    )

    assert actor_prompts == [boundary_only_prompt]
    assert encoded.attention_mask.sum().item() == max_length


def test_rl_prompt_tokenization_retains_maximal_tail_across_bpe_boundaries(
    local_tokenizer,
):
    header = "<|im_start|>user\n"
    content = "jtm2m0ump"
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prompt = header + content + suffix
    max_length = 12
    candidates = [
        header + content[start:] + suffix
        for start in range(len(content) + 1)
    ]
    expected = next(
        candidate
        for candidate in candidates
        if len(
            local_tokenizer(candidate, add_special_tokens=False).input_ids
        )
        <= max_length
    )

    encoded, actor_prompts = trainer_utils.tokenize_rl_prompts(
        local_tokenizer, [prompt], max_length=max_length, device="cpu"
    )

    assert expected == header + "ump" + suffix
    assert actor_prompts == [expected]
    assert encoded.attention_mask.sum().item() <= max_length


def test_rl_prompt_tokenization_does_not_bound_non_byte_bpe_suffix_search():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[UNK]": 0,
        "<|im_start|>user": 1,
        "<|im_end|>": 2,
        "<|im_start|>assistant": 3,
        "<|im_start|>system": 4,
    }
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]"
    )

    header = "<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    newest_content = "prefix " + ("x" * 10_000)
    prompt = (
        "<|im_start|>system\nold context<|im_end|>\n"
        + header
        + newest_content
        + suffix
    )
    expected = header + newest_content[len("prefix") :] + suffix

    assert len(tokenizer(expected, add_special_tokens=False).input_ids) == 3
    assert len(
        tokenizer(
            header + newest_content[len("prefi") :] + suffix,
            add_special_tokens=False,
        ).input_ids
    ) == 4
    assert len(tokenizer(header + newest_content + suffix, add_special_tokens=False).input_ids) == 4

    _, actor_prompts = trainer_utils.tokenize_rl_prompts(
        tokenizer, [prompt], max_length=3, device="cpu"
    )

    assert actor_prompts == [expected]


@pytest.mark.parametrize(
    "rollout_function",
    [train_ppo.collect_rollout, train_grpo.grpo_train_epoch],
)
def test_rl_trainers_score_the_shared_actor_prompt(rollout_function):
    source = inspect.getsource(rollout_function)

    assert "tokenize_rl_prompts(" in source
    assert "calculate_rewards(actor_prompts," in source


@pytest.mark.parametrize("module", [train_ppo, train_grpo])
def test_rl_reward_models_preserve_actor_prompt_content_exactly(module):
    class CapturingRewardModel:
        def __init__(self):
            self.chats = []

        def get_score(self, _tokenizer, chat):
            self.chats.append(chat)
            return 0.0

    reward_model = CapturingRewardModel()
    actor_prompt = (
        "<|im_start|>user\n  retained content tail  <|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    responses = ["reply"] if module is train_ppo else ["reply one", "reply two"]
    args = SimpleNamespace(
        device="cpu", reasoning=0, num_generations=len(responses)
    )

    module.calculate_rewards(
        [actor_prompt], responses, reward_model, reward_tokenizer=None, args=args
    )

    assert reward_model.chats
    for chat in reward_model.chats:
        assert chat[0] == {
            "role": "user",
            "content": "  retained content tail  ",
        }


def test_rollout_masks_use_prompt_layout_and_generated_eos_not_pad_ids():
    prompt_attention_mask = torch.tensor(
        [[0, 1, 1], [1, 1, 0]], dtype=torch.long
    )
    prompt_ids = torch.tensor(
        [[0, 10, 11], [0, 10, 11], [20, 21, 0], [20, 21, 0]]
    )
    completion_ids = torch.tensor(
        [
            [0, 7, 2, 0],
            [5, 2, 0, 0],
            [0, 2, 9, 0],
            [0, 0, 7, 8],
        ]
    )
    generated_ids = torch.cat((prompt_ids, completion_ids), dim=1)

    full_attention_mask, action_mask = trainer_utils.build_rollout_masks(
        prompt_attention_mask, generated_ids, eos_token_id=2
    )

    expected_actions = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    torch.testing.assert_close(action_mask, expected_actions)
    torch.testing.assert_close(
        full_attention_mask[:, :3],
        prompt_attention_mask.repeat_interleave(2, dim=0),
    )
    torch.testing.assert_close(
        full_attention_mask[:, 3:], expected_actions.to(torch.long)
    )


@pytest.mark.parametrize(
    "rollout_function",
    [train_ppo.collect_rollout, train_grpo.grpo_train_epoch],
)
def test_rl_trainers_use_shared_rollout_masks_without_pad_id_comparisons(
    rollout_function,
):
    source = inspect.getsource(rollout_function)

    assert "build_rollout_masks(" in source
    assert "!= tokenizer.pad_token_id" not in source
    assert ".eq(tokenizer.pad_token_id)" not in source


@pytest.mark.parametrize("module", [train_ppo, train_grpo])
def test_sampled_k3_kl_uses_ref_minus_policy_log_probability(module):
    sampled_k3_kl = getattr(module, "sampled_k3_kl", None)
    assert callable(sampled_k3_kl), f"{module.__name__} must expose sampled_k3_kl"

    policy_logp = torch.tensor([math.log(0.5), math.log(0.8)])
    ref_logp = torch.tensor([math.log(0.25), math.log(0.4)])
    d = ref_logp - policy_logp

    torch.testing.assert_close(
        sampled_k3_kl(policy_logp, ref_logp),
        torch.exp(d) - d - 1.0,
    )


@pytest.mark.parametrize("module", [train_ppo, train_grpo])
def test_sampled_k3_kl_extreme_log_ratios_have_finite_values_and_gradients(module):
    policy_logp = torch.tensor([1_000.0, -1_000.0], requires_grad=True)
    ref_logp = torch.tensor([-1_000.0, 1_000.0], requires_grad=True)

    kl = module.sampled_k3_kl(policy_logp, ref_logp)
    kl.sum().backward()

    assert torch.isfinite(kl).all()
    assert torch.isfinite(policy_logp.grad).all()
    assert torch.isfinite(ref_logp.grad).all()


@pytest.mark.parametrize("module", [train_ppo, train_grpo])
def test_clipped_surrogate_extreme_log_ratios_have_finite_values_and_gradients(module):
    clipped = getattr(module, "clipped_surrogate_loss", None)
    assert callable(clipped), f"{module.__name__} must expose clipped_surrogate_loss"
    actor_logps = torch.tensor([-1_000.0, 1_000.0], requires_grad=True)
    old_logps = torch.tensor([1_000.0, -1_000.0])
    advantages = torch.tensor([1.0, -1.0])

    loss, ratio = clipped(
        actor_logps, old_logps, advantages, clip_epsilon=0.2
    )
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert torch.isfinite(ratio).all()
    assert torch.isfinite(actor_logps.grad).all()


def test_ppo_temperature_scales_logprob_and_entropy_distribution():
    helper = getattr(train_ppo, "get_token_logps_and_entropy", None)
    assert callable(helper), "PPO must expose its temperature-aware policy statistics"

    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]]])
    labels = torch.tensor([[0, 2]])
    logps, entropy = helper(logits, labels, temperature=2.0)
    expected_all = torch.log_softmax(logits / 2.0, dim=-1)

    torch.testing.assert_close(logps, expected_all.gather(2, labels.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(entropy, -(expected_all.exp() * expected_all).sum(dim=-1))


def test_grpo_temperature_scales_logprob_and_entropy_distribution():
    assert "temperature" in inspect.signature(train_grpo.get_per_token_logps).parameters

    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0], [1.0, 1.0, 1.0]]]
    )

    class TinyModel:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(logits=logits)

    input_ids = torch.tensor([[2, 0, 2]])
    logps, entropy = train_grpo.get_per_token_logps(
        TinyModel(), input_ids, 2, torch.ones_like(input_ids), temperature=2.0
    )
    expected_all = torch.log_softmax(logits[:, :-1] / 2.0, dim=-1)
    completion_ids = input_ids[:, -2:]

    torch.testing.assert_close(
        logps, expected_all.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(entropy, -(expected_all.exp() * expected_all).sum(dim=-1))


@pytest.mark.parametrize(
    ("path", "option", "expected"),
    [
        ("trainer/train_ppo.py", "--temperature", 0.8),
        ("trainer/train_grpo.py", "--temperature", 0.8),
        ("trainer/train_grpo.py", "--grpo_epochs", 2),
    ],
)
def test_rl_parser_defaults(path, option, expected):
    assert _parser_default(path, option) == expected


@pytest.mark.parametrize("path", ["trainer/train_ppo.py", "trainer/train_grpo.py"])
def test_rl_generation_disables_default_top_k_to_match_full_softmax_stats(path):
    module = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    generate_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]

    assert generate_calls
    assert all(
        any(
            keyword.arg == "top_k"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 0
            for keyword in call.keywords
        )
        for call in generate_calls
    )


@pytest.mark.parametrize("module", [train_ppo, train_grpo])
def test_rl_validation_rejects_nonpositive_temperature_and_float16(module):
    validate_args = getattr(module, "validate_args", None)
    assert callable(validate_args), f"{module.__name__} must expose validate_args"

    kwargs = dict(
        temperature=0.8,
        dtype="bfloat16",
        accumulation_steps=1,
        ppo_epochs=1,
        num_generations=2,
        grpo_epochs=1,
    )
    validate_args(SimpleNamespace(**kwargs))

    with pytest.raises(ValueError, match="temperature"):
        validate_args(SimpleNamespace(**{**kwargs, "temperature": 0.0}))
    with pytest.raises(ValueError, match="temperature"):
        validate_args(SimpleNamespace(**{**kwargs, "temperature": float("nan")}))
    with pytest.raises(ValueError, match="bfloat16"):
        validate_args(SimpleNamespace(**{**kwargs, "dtype": "float16"}))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("num_generations", 1, "num_generations"), ("grpo_epochs", 0, "grpo_epochs")],
)
def test_grpo_validation_rejects_invalid_group_and_update_counts(field, value, message):
    validate_args = getattr(train_grpo, "validate_args", None)
    assert callable(validate_args), "GRPO must expose validate_args"
    kwargs = dict(
        temperature=0.8,
        dtype="bfloat16",
        accumulation_steps=1,
        num_generations=2,
        grpo_epochs=2,
    )

    with pytest.raises(ValueError, match=message):
        validate_args(SimpleNamespace(**{**kwargs, field: value}))


def test_grpo_requires_single_step_accumulation_for_multi_epoch_updates():
    validate_args = getattr(train_grpo, "validate_args", None)
    assert callable(validate_args), "GRPO must expose validate_args"

    with pytest.raises(ValueError, match="accumulation_steps=1"):
        validate_args(
            SimpleNamespace(
                temperature=0.8,
                dtype="bfloat16",
                accumulation_steps=2,
                num_generations=2,
                grpo_epochs=2,
            )
        )


def test_ppo_early_stop_is_synchronized_across_ranks(monkeypatch):
    synchronize = getattr(train_ppo, "synchronize_early_stop", None)
    assert callable(synchronize), "PPO must expose synchronize_early_stop"
    calls = []

    monkeypatch.setattr(train_ppo.dist, "is_initialized", lambda: True)

    def remote_rank_requests_stop(flag, op):
        calls.append(op)
        flag.fill_(1)

    monkeypatch.setattr(train_ppo.dist, "all_reduce", remote_rank_requests_stop)

    assert synchronize(False, torch.device("cpu")) is True
    assert calls == [train_ppo.dist.ReduceOp.MAX]
    assert "synchronize_early_stop" in inspect.getsource(train_ppo.ppo_update)


def test_ppo_target_kl_estimates_old_policy_to_current_actor():
    class TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(math.log(4.0)))

        def forward(self, input_ids, attention_mask):
            selected_logits = self.logit.expand(*input_ids.shape)
            logits = torch.stack(
                (torch.zeros_like(selected_logits), selected_logits), dim=-1
            )
            return SimpleNamespace(logits=logits)

    class TinyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids, attention_mask):
            return self.value.expand(*input_ids.shape)

    class CountingScheduler:
        def step(self):
            pass

    actor = TinyActor()
    critic = TinyCritic()
    current_logp = torch.log(torch.tensor(0.8))
    rollout = {
        "gen_out": torch.tensor([[0, 1]]),
        "full_mask": torch.ones(1, 2, dtype=torch.long),
        "labels": torch.tensor([[1]]),
        "resp_mask": torch.ones(1, 1, dtype=torch.bool),
        "old_logp": (current_logp - 1.0).reshape(1, 1),
        "ref_logp": current_logp.reshape(1, 1),
        "advantages": torch.ones(1, 1),
        "returns": torch.zeros(1, 1),
        "old_values": torch.zeros(1, 1),
        "rewards": torch.zeros(1),
    }
    args = SimpleNamespace(
        ppo_epochs=2,
        device="cpu",
        temperature=1.0,
        clip_epsilon=0.2,
        target_kl=0.5,
        vf_coef=0.0,
        kl_coef=0.0,
        entropy_coef=0.0,
        grad_clip=0.0,
    )

    metrics = train_ppo.ppo_update(
        rollout,
        actor,
        critic,
        torch.optim.SGD(actor.parameters(), lr=0.0),
        torch.optim.SGD(critic.parameters(), lr=0.0),
        CountingScheduler(),
        CountingScheduler(),
        tokenizer=None,
        args=args,
        autocast_ctx=nullcontext(),
        lm_config=SimpleNamespace(use_moe=False),
    )

    assert metrics["ppo_epochs_actual"] == 1


def test_grpo_uses_frozen_rollout_logps_for_multiple_clipped_updates():
    clipped = getattr(train_grpo, "clipped_surrogate_loss", None)
    assert callable(clipped), "GRPO must expose clipped_surrogate_loss"

    new_logps = torch.log(torch.tensor([[1.5, 0.5]]))
    old_logps = torch.zeros_like(new_logps)
    advantages = torch.tensor([[1.0, -1.0]])
    loss, ratio = clipped(new_logps, old_logps, advantages, clip_epsilon=0.2)

    torch.testing.assert_close(ratio, torch.tensor([[1.5, 0.5]]))
    torch.testing.assert_close(loss, torch.tensor([[-1.2, 0.8]]))

    source = inspect.getsource(train_grpo.grpo_train_epoch)
    assert "range(args.grpo_epochs)" in source
    assert "clipped_surrogate_loss" in source
    old_snapshot = source.index("old_logps")
    assert source.rfind("with torch.no_grad()", 0, old_snapshot) > source.index("outputs =")


def test_grpo_executes_multiple_updates_against_one_frozen_rollout(monkeypatch):
    class BatchEncoding(dict):
        def to(self, _device):
            return self

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, prompts, **_kwargs):
            return BatchEncoding(
                input_ids=torch.ones(len(prompts), 1, dtype=torch.long),
                attention_mask=torch.ones(len(prompts), 1, dtype=torch.long),
            )

        def batch_decode(self, completion_ids, **_kwargs):
            return [str(row.tolist()) for row in completion_ids]

    class TinyPolicy(torch.nn.Module):
        def __init__(self, value=0.0):
            super().__init__()
            self.preference = torch.nn.Parameter(torch.tensor(value))
            self.forward_calls = 0

        def generate(self, **_kwargs):
            return torch.tensor([[1, 1], [1, 2]], dtype=torch.long)

        def forward(self, input_ids, **_kwargs):
            self.forward_calls += 1
            batch, length = input_ids.shape
            logits = torch.zeros(batch, length, 3)
            logits[..., 1] = self.preference
            logits[..., 2] = -self.preference
            return SimpleNamespace(logits=logits)

    class CountingScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    policy = TinyPolicy()
    reference = TinyPolicy()
    reference.requires_grad_(False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    scheduler = CountingScheduler()
    observed_ratios = []
    original_clipped = train_grpo.clipped_surrogate_loss

    def recording_clipped(actor_logps, old_logps, advantages, clip_epsilon):
        assert old_logps.requires_grad is False
        loss, ratio = original_clipped(
            actor_logps, old_logps, advantages, clip_epsilon
        )
        observed_ratios.append(ratio.detach().clone())
        return loss, ratio

    monkeypatch.setattr(train_grpo, "model", policy, raising=False)
    monkeypatch.setattr(train_grpo, "tokenizer", TinyTokenizer(), raising=False)
    monkeypatch.setattr(train_grpo, "optimizer", optimizer, raising=False)
    monkeypatch.setattr(train_grpo, "scheduler", scheduler, raising=False)
    monkeypatch.setattr(train_grpo, "autocast_ctx", nullcontext(), raising=False)
    monkeypatch.setattr(train_grpo, "lm_config", SimpleNamespace(use_moe=False), raising=False)
    monkeypatch.setattr(
        train_grpo,
        "args",
        SimpleNamespace(
            device="cpu",
            max_seq_len=0,
            max_gen_len=1,
            temperature=0.8,
            num_generations=2,
            clip_epsilon=0.2,
            beta=0.0,
            entropy_coef=0.0,
            grad_clip=0.0,
            grpo_epochs=2,
            log_interval=10,
            epochs=1,
            accumulation_steps=1,
            save_interval=10,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        train_grpo,
        "calculate_rewards",
        lambda *_args, **_kwargs: torch.tensor([1.0, -1.0]),
    )
    monkeypatch.setattr(train_grpo, "checkpoint_due", lambda *_args: False)
    monkeypatch.setattr(train_grpo, "clipped_surrogate_loss", recording_clipped)

    train_grpo.grpo_train_epoch(
        0,
        [{"prompt": ["prompt"]}],
        1,
        reference,
        reward_model=None,
        reward_tokenizer=None,
    )

    assert policy.forward_calls == 3  # rollout snapshot + two fresh update forwards
    assert scheduler.steps == 2
    torch.testing.assert_close(observed_ratios[0], torch.ones_like(observed_ratios[0]))
    assert not torch.allclose(observed_ratios[1], torch.ones_like(observed_ratios[1]))


@pytest.mark.parametrize("path", ["trainer/train_ppo.py", "trainer/train_grpo.py"])
def test_rl_trainers_have_no_float16_execution_path(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    assert "torch.float16" not in source


@pytest.mark.parametrize(
    ("path", "train_call"),
    [
        ("trainer/train_ppo.py", "ppo_train_epoch("),
        ("trainer/train_grpo.py", "grpo_train_epoch("),
    ],
)
def test_rl_resume_restores_rng_after_epoch_setup_and_loader_creation(
    path, train_call
):
    source = (ROOT / path).read_text(encoding="utf-8")
    loop = source[source.index("for epoch in range(start_epoch, args.epochs):") :]

    assert loop.index(
        "setup_seed(42 + epoch * world_size + rank)"
    ) < loop.index("DataLoader(")
    assert loop.index("DataLoader(") < loop.index("restore_rng_state_for_rank(")
    assert loop.index("restore_rng_state_for_rank(") < loop.index(train_call)
    assert "generator=loader_generator" in loop
    assert "train_ds.set_epoch(epoch)" in loop


@pytest.mark.parametrize("path", ["trainer/train_ppo.py", "trainer/train_grpo.py"])
def test_rl_checkpoint_rng_collectives_are_not_main_rank_only(path):
    module = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    parents = {
        child: parent
        for parent in ast.walk(module)
        for child in ast.iter_child_nodes(parent)
    }
    gather_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gather_rng_states"
    ]
    save_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "lm_checkpoint"
        and any(keyword.arg == "model" for keyword in node.keywords)
    ]

    assert len(gather_calls) >= 2
    assert save_calls
    assert all(
        any(keyword.arg == "rng_state_by_rank" for keyword in call.keywords)
        for call in save_calls
    )
    for call in gather_calls:
        ancestor = parents.get(call)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                assert not any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "is_main_process"
                    for node in ast.walk(ancestor.test)
                )
            ancestor = parents.get(ancestor)


def test_rlaif_prompt_processing_is_deterministic_by_epoch_and_index(
    tmp_path, monkeypatch
):
    class TinyTokenizer:
        pad_token_id = 0

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_ids=[1])

        def apply_chat_template(self, *_args, **_kwargs):
            return "<think>\n\n</think>\n\nquestion"

    data_path = tmp_path / "rl.jsonl"
    data_path.write_text(
        '{"conversations": [{"content": "q"}, {"content": "a"}]}\n',
        encoding="utf-8",
    )
    dataset = llm_dataset.RLAIFDataset(
        data_path, TinyTokenizer(), max_length=32, seed=17
    )
    dataset.set_epoch(3)
    global_draws = iter([0.0, 1.0])
    monkeypatch.setattr(llm_dataset.random, "random", lambda: next(global_draws))

    assert dataset[0] == dataset[0]


@pytest.mark.parametrize("path", ["trainer/train_ppo.py", "trainer/train_grpo.py"])
def test_rl_trainers_use_shared_nonfinite_gradient_guard(path):
    module = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "clip_gradients"
    ]

    assert calls


@pytest.mark.parametrize("path", ["trainer/train_ppo.py", "trainer/train_grpo.py"])
def test_rl_epoch_seed_includes_rank_for_distributed_rollout_diversity(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    loop = source[source.index("for epoch in range(start_epoch, args.epochs):") :]

    assert "setup_seed(42 + epoch * world_size + rank)" in loop


@pytest.mark.parametrize(
    "path",
    ["trainer/train_dpo.py", "trainer/train_ppo.py", "trainer/train_grpo.py"],
)
def test_preference_trainers_checkpoint_and_restore_frozen_reference(path):
    module = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    save_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "lm_checkpoint"
        and any(keyword.arg == "model" for keyword in node.keywords)
    ]
    restore_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "init_reference_model"
    ]
    sync_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "synchronize_model_state"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ref_model"
    ]

    assert save_calls
    assert all(
        any(keyword.arg == "ref_model" for keyword in call.keywords)
        for call in save_calls
    )
    assert restore_calls
    assert sync_calls


@pytest.mark.parametrize("path", ["trainer/train_ppo.py", "trainer/train_grpo.py"])
def test_rl_trainers_load_reward_components_through_temporary_cache(path):
    module = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_reward_components"
    ]

    assert calls
