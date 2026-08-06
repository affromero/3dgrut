"""Temporary restoration of checkpoint dataset membership rules."""

from collections.abc import Iterator
from contextlib import contextmanager


def dataset_membership(conf: object) -> dict[str, object]:
    """Snapshot fields that determine train and validation membership."""
    dataset = conf.dataset
    return {
        "holdout_image_list_path": dataset.get("holdout_image_list_path"),
        "train_exclude_image_list_path": dataset.get(
            "train_exclude_image_list_path"
        ),
        "test_split_interval": int(dataset.test_split_interval),
    }


@contextmanager
def use_dataset_membership(
    conf: object,
    membership: dict[str, object],
) -> Iterator[None]:
    """Restore one membership snapshot for the lifetime of a renderer."""
    previous = dataset_membership(conf)
    required = {
        "holdout_image_list_path",
        "train_exclude_image_list_path",
        "test_split_interval",
    }
    if set(membership) != required:
        message = "dataset membership snapshot is incomplete"
        raise ValueError(message)
    try:
        for key, value in membership.items():
            setattr(conf.dataset, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(conf.dataset, key, value)


def training_membership_provenance(
    membership: dict[str, object],
) -> dict[str, object]:
    """Describe the frozen checkpoint rule used for training support."""
    holdout = membership["holdout_image_list_path"]
    interval = int(membership["test_split_interval"])
    if holdout:
        source = "checkpoint_holdout_image_list"
    elif interval > 0:
        source = "checkpoint_test_split_interval"
    else:
        source = "checkpoint_all_registered_images"
    return {"source": source, **membership}
