"""Runtime contracts for offline dynamic padding and length-aware batching."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from dataset.llm_dataset import (
    DPODataset,
    PretrainDataset,
    SFTDataset,
    dynamic_padding_collate,
)
from trainer.train_dpo import align_dpo_branches_for_forward
from trainer.trainer_utils import build_epoch_batch_sampler


def _tuple_sample(offset, effective_length, total_length=7):
    values = torch.arange(offset, offset + total_length)
    attention_mask = torch.tensor(
        [1] * effective_length + [0] * (total_length - effective_length),
        dtype=torch.long,
    )
    loss_mask = attention_mask.clone()
    return values, values + 100, loss_mask, attention_mask


def _dpo_sample(offset, chosen_length, rejected_length, total_length=8):
    def branch(branch_name, prefix, effective_length):
        values = torch.arange(prefix, prefix + total_length)
        attention_mask = torch.tensor(
            [1] * effective_length + [0] * (total_length - effective_length),
            dtype=torch.long,
        )
        return {
            f"x_{branch_name}": values,
            f"y_{branch_name}": values + 100,
            f"mask_{branch_name}": attention_mask.clone(),
            f"attention_mask_{branch_name}": attention_mask,
        }

    chosen = branch("chosen", offset, chosen_length)
    rejected = branch("rejected", offset + 1000, rejected_length)
    return {**chosen, **rejected}


class _LengthProbeTokenizer:
    pad_token_id = 0
    _special_tokens = {
        "<|im_start|>assistant\n": [1],
        "<|im_end|>\n": [2],
    }

    def __call__(self, text, **kwargs):
        if text in self._special_tokens:
            return SimpleNamespace(input_ids=self._special_tokens[text])
        raise AssertionError("bucket-length initialization must not tokenize raw data")

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("bucket-length initialization must not render ChatML")


def test_bucket_length_initialization_uses_raw_content_without_tokenizing_or_rendering(tmp_path):
    tokenizer = _LengthProbeTokenizer()
    pretrain_path = tmp_path / "pretrain.jsonl"
    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    pretrain_text = "p" * 31
    sft_contents = ("s" * 17, "t" * 29)
    chosen_content = "c" * 41
    rejected_content = "r" * 53

    pretrain_path.write_text(json.dumps({"text": pretrain_text}) + "\n", encoding="utf-8")
    sft_path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": sft_contents[0]},
                    {"role": "assistant", "content": sft_contents[1]},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dpo_path.write_text(
        json.dumps(
            {
                "chosen": [{"role": "assistant", "content": chosen_content}],
                "rejected": [{"role": "assistant", "content": rejected_content}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pretrain = PretrainDataset(pretrain_path, tokenizer, max_length=8)
    sft = SFTDataset(sft_path, tokenizer, max_length=8)
    dpo = DPODataset(dpo_path, tokenizer, max_length=8)

    assert pretrain.lengths == [len(pretrain_text)]
    assert sft.lengths == [sum(map(len, sft_contents))]
    assert dpo.lengths == [len(rejected_content)]


def test_dynamic_padding_tuple_collate_crops_to_batch_max_without_changing_valid_tokens():
    short = _tuple_sample(10, effective_length=3)
    long = _tuple_sample(30, effective_length=5)

    X, Y, loss_mask, attention_mask = dynamic_padding_collate([short, long])

    assert [tensor.shape for tensor in (X, Y, loss_mask, attention_mask)] == [
        (2, 5),
        (2, 5),
        (2, 5),
        (2, 5),
    ]
    for row, sample in enumerate((short, long)):
        for output, original in zip((X, Y, loss_mask, attention_mask), sample):
            assert torch.equal(output[row], original[:5])
    assert attention_mask.sum().item() == short[3].sum().item() + long[3].sum().item()
    assert loss_mask.sum().item() == short[2].sum().item() + long[2].sum().item()


def test_dynamic_padding_dpo_collate_crops_chosen_and_rejected_independently():
    first = _dpo_sample(10, chosen_length=2, rejected_length=6)
    second = _dpo_sample(30, chosen_length=4, rejected_length=3)

    batch = dynamic_padding_collate([first, second])

    assert batch["x_chosen"].shape == (2, 4)
    assert batch["x_rejected"].shape == (2, 6)
    for key in ("x_chosen", "y_chosen", "mask_chosen", "attention_mask_chosen"):
        assert torch.equal(batch[key][0], first[key][:4])
        assert torch.equal(batch[key][1], second[key][:4])
    for key in ("x_rejected", "y_rejected", "mask_rejected", "attention_mask_rejected"):
        assert torch.equal(batch[key][0], first[key][:6])
        assert torch.equal(batch[key][1], second[key][:6])
    for branch in ("chosen", "rejected"):
        assert batch[f"attention_mask_{branch}"].sum().item() == sum(
            sample[f"attention_mask_{branch}"].sum().item()
            for sample in (first, second)
        )
        assert batch[f"mask_{branch}"].sum().item() == sum(
            sample[f"mask_{branch}"].sum().item() for sample in (first, second)
        )


def test_dpo_forward_alignment_pads_only_the_shorter_cropped_branch_on_the_right():
    first = _dpo_sample(10, chosen_length=2, rejected_length=6)
    second = _dpo_sample(30, chosen_length=4, rejected_length=3)
    cropped = dynamic_padding_collate([first, second])

    x, y, loss_mask, attention_mask = align_dpo_branches_for_forward(
        cropped["x_chosen"],
        cropped["y_chosen"],
        cropped["mask_chosen"],
        cropped["attention_mask_chosen"],
        cropped["x_rejected"],
        cropped["y_rejected"],
        cropped["mask_rejected"],
        cropped["attention_mask_rejected"],
    )

    assert [tensor.shape for tensor in (x, y, loss_mask, attention_mask)] == [
        (4, 6),
        (4, 6),
        (4, 6),
        (4, 6),
    ]
    assert torch.equal(x[:2, :4], cropped["x_chosen"])
    assert torch.equal(y[:2, :4], cropped["y_chosen"])
    assert torch.equal(loss_mask[:2, :4], cropped["mask_chosen"])
    assert torch.equal(attention_mask[:2, :4], cropped["attention_mask_chosen"])
    assert torch.equal(x[2:], cropped["x_rejected"])
    assert torch.equal(y[2:], cropped["y_rejected"])
    assert torch.equal(loss_mask[2:], cropped["mask_rejected"])
    assert torch.equal(attention_mask[2:], cropped["attention_mask_rejected"])
    assert torch.equal(x[:2, 4:], torch.zeros((2, 2), dtype=x.dtype))
    assert torch.equal(y[:2, 4:], torch.zeros((2, 2), dtype=y.dtype))
    assert torch.equal(loss_mask[:2, 4:], torch.zeros((2, 2), dtype=loss_mask.dtype))
    assert torch.equal(
        attention_mask[:2, 4:], torch.zeros((2, 2), dtype=attention_mask.dtype)
    )


def test_length_bucket_batches_are_locally_grouped_and_resume_is_a_suffix():
    indices = list(range(8))
    lengths = [8, 1, 7, 2, 6, 3, 5, 4]
    normal_batches = list(
        build_epoch_batch_sampler(
            dataset_size=len(indices),
            batch_size=2,
            epoch=0,
            sampler=indices,
            lengths=lengths,
            bucket_window_multiplier=2,
        )
    )
    resumed_batches = list(
        build_epoch_batch_sampler(
            dataset_size=len(indices),
            batch_size=2,
            epoch=0,
            skip_batches=1,
            sampler=indices,
            lengths=lengths,
            bucket_window_multiplier=2,
        )
    )

    assert normal_batches == [[1, 3], [2, 0], [5, 7], [6, 4]]
    assert resumed_batches == normal_batches[1:]
    assert all(
        lengths[batch[0]] <= lengths[batch[1]] for batch in normal_batches
    )
    globally_sorted = sorted(indices, key=lambda index: lengths[index])
    assert normal_batches != [globally_sorted[i:i + 2] for i in range(0, 8, 2)]


def test_all_offline_trainers_wire_dynamic_collate_and_lengths_into_batching():
    project_root = Path(__file__).resolve().parents[1]
    trainer_dir = project_root / "trainer"
    expected = {
        "train_pretrain.py",
        "train_full_sft.py",
        "train_lora.py",
        "train_dpo.py",
        "train_reason.py",
    }
    wired_collate = set()
    wired_lengths = set()

    for trainer_path in trainer_dir.glob("train_*.py"):
        source = trainer_path.read_text(encoding="utf-8").replace(" ", "").replace("\n", "")
        if "collate_fn=dynamic_padding_collate" in source:
            wired_collate.add(trainer_path.name)
        if "lengths=train_ds.lengths" in source:
            wired_lengths.add(trainer_path.name)

    assert wired_collate == expected
    assert wired_lengths == expected


@pytest.fixture(scope="module")
def local_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizer_path = Path(__file__).resolve().parents[1] / "model"
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path)


def _write_jsonl(path, row):
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_sft_long_prompt_preserves_supervised_answer_and_terminal_eos(
    tmp_path, local_tokenizer
):
    path = tmp_path / "long-sft.jsonl"
    _write_jsonl(
        path,
        {
            "conversations": [
                {"role": "user", "content": "very long prompt " * 100},
                {"role": "assistant", "content": "the retained answer"},
            ]
        },
    )

    _, labels, loss_mask, _ = SFTDataset(
        path, local_tokenizer, max_length=64
    )[0]
    supervised_labels = labels[loss_mask.bool()]

    assert supervised_labels.numel() > 0
    assert local_tokenizer.eos_token_id in supervised_labels.tolist()


def test_sft_truncation_starts_at_complete_chatml_message_boundary(
    tmp_path, local_tokenizer
):
    path = tmp_path / "chatml-boundary-sft.jsonl"
    _write_jsonl(
        path,
        {
            "conversations": [
                {
                    "role": "user",
                    "content": (
                        "HEAD_SENTINEL "
                        + "overlong user context " * 100
                        + " USER_TAIL_SENTINEL"
                    ),
                },
                {"role": "assistant", "content": "retained answer"},
            ]
        },
    )

    inputs, _, _, attention_mask = SFTDataset(
        path, local_tokenizer, max_length=48
    )[0]
    valid_ids = inputs[attention_mask.bool()].tolist()
    user_header = local_tokenizer(
        "<|im_start|>user\n", add_special_tokens=False
    ).input_ids
    assistant_header = local_tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    ).input_ids
    eos_boundary = local_tokenizer(
        "<|im_end|>\n", add_special_tokens=False
    ).input_ids

    assert valid_ids[:len(user_header)] == user_header
    assert _find_subsequence(valid_ids, assistant_header) > 0
    assert _find_subsequence(valid_ids, eos_boundary) >= len(user_header)
    _assert_chatml_starts_are_bounded(
        valid_ids, assistant_header[:1], eos_boundary
    )
    assert "USER_TAIL_SENTINEL" in local_tokenizer.decode(
        valid_ids, skip_special_tokens=False
    )


def test_sft_overlong_final_assistant_preserves_content_tail_and_boundaries(
    tmp_path, local_tokenizer
):
    path = tmp_path / "overlong-assistant-sft.jsonl"
    _write_jsonl(
        path,
        {
            "conversations": [
                {"role": "user", "content": "brief question"},
                {
                    "role": "assistant",
                    "content": "HEAD_SENTINEL " + "filler " * 100 + "TAIL_SENTINEL",
                },
            ]
        },
    )

    inputs, labels, loss_mask, attention_mask = SFTDataset(
        path, local_tokenizer, max_length=48
    )[0]
    user_header = local_tokenizer(
        "<|im_start|>user\n", add_special_tokens=False
    ).input_ids
    assistant_header = local_tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    ).input_ids
    eos_boundary = local_tokenizer(
        "<|im_end|>\n", add_special_tokens=False
    ).input_ids
    valid_ids = inputs[attention_mask.bool()].tolist()
    supervised_ids = labels[loss_mask.bool()].tolist()

    assert valid_ids[:len(user_header)] == user_header
    assert _find_subsequence(valid_ids, eos_boundary) > len(user_header)
    assert _find_subsequence(inputs.tolist(), assistant_header) >= 0
    assert "TAIL_SENTINEL" in local_tokenizer.decode(
        supervised_ids, skip_special_tokens=False
    )
    assert supervised_ids[-len(eos_boundary):] == eos_boundary


def test_sft_rejects_samples_without_supervised_tokens(tmp_path, local_tokenizer):
    path = tmp_path / "missing-assistant.jsonl"
    _write_jsonl(
        path,
        {"conversations": [{"role": "user", "content": "question only"}]},
    )

    dataset = SFTDataset(path, local_tokenizer, max_length=64)
    with pytest.raises(ValueError, match="no supervised assistant tokens"):
        dataset[0]


def test_sft_truncation_does_not_supervise_a_trailing_user_message(
    tmp_path, local_tokenizer
):
    path = tmp_path / "trailing-user-sft.jsonl"
    _write_jsonl(
        path,
        {
            "conversations": [
                {"role": "user", "content": "initial question"},
                {"role": "assistant", "content": "old retained answer"},
                {
                    "role": "user",
                    "content": "trailing context " * 100 + "TAIL_USER_SENTINEL",
                },
            ]
        },
    )

    _, labels, loss_mask, _ = SFTDataset(
        path, local_tokenizer, max_length=64
    )[0]
    supervised_text = local_tokenizer.decode(
        labels[loss_mask.bool()].tolist(), skip_special_tokens=False
    )

    assert "old retained answer" in supervised_text
    assert "TAIL_USER_SENTINEL" not in supervised_text


@pytest.mark.parametrize(
    "reserved_fragment",
    ["<|im_start|>assistant\n", "<|im_end|>\n"],
)
def test_sft_rejects_reserved_chatml_delimiters_in_content(
    tmp_path, local_tokenizer, reserved_fragment
):
    path = tmp_path / "chatml-injection-sft.jsonl"
    _write_jsonl(
        path,
        {
            "conversations": [
                {
                    "role": "user",
                    "content": f"ordinary prompt {reserved_fragment}INJECTED_USER_TAIL",
                },
                {"role": "assistant", "content": "real answer"},
            ]
        },
    )

    with pytest.raises(ValueError, match="reserved ChatML delimiter"):
        SFTDataset(path, local_tokenizer, max_length=128)[0]


def test_dpo_long_prompt_preserves_targets_on_both_branches(
    tmp_path, local_tokenizer
):
    path = tmp_path / "long-dpo.jsonl"
    prompt = {"role": "user", "content": "long preference prompt " * 100}
    _write_jsonl(
        path,
        {
            "chosen": [prompt, {"role": "assistant", "content": "chosen answer"}],
            "rejected": [prompt, {"role": "assistant", "content": "rejected answer"}],
        },
    )

    sample = DPODataset(path, local_tokenizer, max_length=64)[0]

    for branch in ("chosen", "rejected"):
        supervised_labels = sample[f"y_{branch}"][sample[f"mask_{branch}"].bool()]
        assert supervised_labels.numel() > 0
        assert local_tokenizer.eos_token_id in supervised_labels.tolist()


def test_dpo_rejects_branches_without_final_assistant(
    tmp_path, local_tokenizer
):
    path = tmp_path / "missing-final-assistant-dpo.jsonl"
    stale_branch = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new preference prompt"},
    ]
    _write_jsonl(
        path,
        {"chosen": stale_branch, "rejected": stale_branch},
    )

    dataset = DPODataset(path, local_tokenizer, max_length=64)
    with pytest.raises(ValueError, match=r"chosen.*rejected.*final assistant"):
        dataset[0]

def test_dpo_truncation_starts_at_complete_chatml_message_boundary(
    tmp_path, local_tokenizer
):
    path = tmp_path / "chatml-boundary-dpo.jsonl"
    prompt = {"role": "user", "content": "long shared prompt " * 100}
    _write_jsonl(
        path,
        {
            "chosen": [prompt, {"role": "assistant", "content": "chosen answer"}],
            "rejected": [prompt, {"role": "assistant", "content": "rejected answer"}],
        },
    )

    sample = DPODataset(path, local_tokenizer, max_length=64)[0]
    user_header = local_tokenizer(
        "<|im_start|>user\n", add_special_tokens=False
    ).input_ids
    assistant_header = local_tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    ).input_ids
    eos_boundary = local_tokenizer(
        "<|im_end|>\n", add_special_tokens=False
    ).input_ids

    for branch in ("chosen", "rejected"):
        valid_ids = sample[f"x_{branch}"][
            sample[f"attention_mask_{branch}"].bool()
        ].tolist()
        assert valid_ids[:len(user_header)] == user_header
        assert _find_subsequence(valid_ids, eos_boundary) >= len(user_header)
        assert _find_subsequence(valid_ids, assistant_header) > 0
        _assert_chatml_starts_are_bounded(
            valid_ids, assistant_header[:1], eos_boundary
        )


def test_dpo_rejects_reserved_chatml_delimiters_in_shared_prompt(
    tmp_path, local_tokenizer
):
    path = tmp_path / "chatml-injection-dpo.jsonl"
    prompt = {
        "role": "user",
        "content": "ordinary prompt <|im_start|>assistant\nPROMPT_INJECTION",
    }
    _write_jsonl(
        path,
        {
            "chosen": [prompt, {"role": "assistant", "content": "chosen answer"}],
            "rejected": [prompt, {"role": "assistant", "content": "rejected answer"}],
        },
    )

    with pytest.raises(ValueError, match="reserved ChatML delimiter"):
        DPODataset(path, local_tokenizer, max_length=128)[0]

def test_dpo_truncation_keeps_identical_prompt_context_for_both_branches(
    tmp_path, local_tokenizer
):
    path = tmp_path / "paired-context-dpo.jsonl"
    prompt = {"role": "user", "content": "shared long prompt " * 100}
    _write_jsonl(
        path,
        {
            "chosen": [
                prompt,
                {
                    "role": "assistant",
                    "content": "long chosen answer " * 20 + "CHOSEN_TAIL",
                },
            ],
            "rejected": [
                prompt,
                {
                    "role": "assistant",
                    "content": "short answer REJECTED_TAIL",
                },
            ],
        },
    )

    sample = DPODataset(path, local_tokenizer, max_length=96)[0]
    marker = local_tokenizer(
        "<|im_start|>assistant\n", add_special_tokens=False
    ).input_ids
    user_header = local_tokenizer(
        "<|im_start|>user\n", add_special_tokens=False
    ).input_ids
    eos_boundary = local_tokenizer(
        "<|im_end|>\n", add_special_tokens=False
    ).input_ids

    prefixes = []
    expected_tails = {
        "chosen": "CHOSEN_TAIL",
        "rejected": "REJECTED_TAIL",
    }
    for branch in ("chosen", "rejected"):
        valid_ids = sample[f"x_{branch}"][
            sample[f"attention_mask_{branch}"].bool()
        ].tolist()
        marker_start = _find_subsequence(valid_ids, marker)
        prefixes.append(valid_ids[:marker_start])
        supervised_ids = sample[f"y_{branch}"][
            sample[f"mask_{branch}"].bool()
        ].tolist()
        assert expected_tails[branch] in local_tokenizer.decode(
            supervised_ids, skip_special_tokens=False
        )
        assert supervised_ids[-len(eos_boundary):] == eos_boundary
        _assert_chatml_starts_are_bounded(
            valid_ids, marker[:1], eos_boundary
        )

    assert prefixes[0] == prefixes[1]
    assert prefixes[0]
    assert _find_subsequence(prefixes[0], eos_boundary) > len(user_header)
    assert sample["attention_mask_chosen"].sum().item() == 95


def test_pretrain_appends_terminal_eos_before_padding(tmp_path, local_tokenizer):
    path = tmp_path / "raw-pretrain.jsonl"
    _write_jsonl(path, {"text": "plain document without a boundary"})

    _, labels, loss_mask, _ = PretrainDataset(
        path, local_tokenizer, max_length=16
    )[0]
    supervised_labels = labels[loss_mask.bool()]

    assert supervised_labels[-1].item() == local_tokenizer.eos_token_id


@pytest.mark.parametrize("max_length", [0, 1])
def test_pretrain_rejects_max_length_that_cannot_form_x_and_y(
    tmp_path, local_tokenizer, max_length
):
    path = tmp_path / "too-short-pretrain.jsonl"
    _write_jsonl(path, {"text": "plain document"})

    with pytest.raises(ValueError, match="max_length"):
        PretrainDataset(path, local_tokenizer, max_length=max_length)


def _find_subsequence(sequence, pattern):
    for index in range(len(sequence) - len(pattern) + 1):
        if sequence[index:index + len(pattern)] == pattern:
            return index
    raise AssertionError(f"pattern {pattern!r} not found")


def _assert_chatml_starts_are_bounded(sequence, message_start, eos_boundary):
    starts = 0
    for index in range(len(sequence) - len(message_start) + 1):
        if sequence[index:index + len(message_start)] != message_start:
            continue
        starts += 1
        assert index == 0 or (
            index >= len(eos_boundary)
            and sequence[index - len(eos_boundary):index] == eos_boundary
        )
    assert starts > 0
