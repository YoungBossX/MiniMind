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
