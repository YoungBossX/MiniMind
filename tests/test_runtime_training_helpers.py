import pytest


torch = pytest.importorskip("torch")

from trainer import trainer_utils


def test_epoch_batch_sampler_resume_is_suffix_of_normal_epoch_stream():
    """A resumed epoch must consume the same remaining batches as an uninterrupted run."""
    normal_batches = list(
        trainer_utils.build_epoch_batch_sampler(
            dataset_size=11,
            batch_size=3,
            epoch=4,
            skip_batches=0,
        )
    )
    resumed_batches = list(
        trainer_utils.build_epoch_batch_sampler(
            dataset_size=11,
            batch_size=3,
            epoch=4,
            skip_batches=2,
        )
    )

    assert resumed_batches == normal_batches[2:]
