"""Vectorized, memory-mapped access to minimax feature records."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import HEADER, RECORD_BYTES, FeatureArtifact, open_feature_artifact


SPLITS = {"train": 1, "validation": 2, "test": 3}
TERMINAL_FLAG = 1 << 0
PARITY_BACKOFF_FLAG = 1 << 1

RECORD_DTYPE = np.dtype(
    {
        "names": [
            "player_one",
            "player_two",
            "key_low",
            "key_high",
            "source_id",
            "family_id",
            "trajectory_id",
            "parent_id",
            "trajectory_index",
            "nodes",
            "completed_nodes",
            "deepest_completed_nodes",
            "previous_completed_nodes",
            "source_shard",
            "source_ordinal",
            "teacher_value",
            "deepest_value",
            "previous_value",
            "normalized_value",
            "two_ply_closure_value",
            "residual",
            "duplicate_count",
            "policy_id",
            "sample_index",
            "ply",
            "side_to_move",
            "split",
            "mode",
            "completed_depth",
            "deepest_completed_depth",
            "previous_completed_depth",
            "attempted_depth",
            "terminal_depth",
            "best_move",
            "deepest_best_move",
            "previous_best_move",
            "flags",
            "reserved",
            "line_patterns",
            "own_gains",
            "opponent_gains",
            "trailing_reserved",
        ],
        "formats": [
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u8",
            "<u4",
            "<u4",
            "<i4",
            "<i4",
            "<i4",
            "<i4",
            "<i4",
            "<i4",
            "<u4",
            "<u2",
            "<u2",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            "u1",
            ("u1", 3),
            ("<u2", 36),
            ("<i2", 49),
            ("<i2", 49),
            "<u4",
        ],
        "offsets": [
            0,
            8,
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            108,
            112,
            116,
            120,
            124,
            128,
            132,
            136,
            140,
            142,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            160,
            232,
            330,
            428,
        ],
        "itemsize": RECORD_BYTES,
    }
)
assert RECORD_DTYPE.itemsize == RECORD_BYTES


@dataclass(frozen=True)
class NumpyFeatureBatch:
    """Contiguous arrays materialized in one bulk operation."""

    indices: np.ndarray
    line_patterns: np.ndarray
    own_gains: np.ndarray
    opponent_gains: np.ndarray
    teacher_values: np.ndarray
    two_ply_closure_values: np.ndarray
    residuals: np.ndarray
    plys: np.ndarray
    completed_depths: np.ndarray
    deepest_completed_depths: np.ndarray
    terminal_depths: np.ndarray
    flags: np.ndarray
    duplicate_counts: np.ndarray
    family_ids: np.ndarray
    trajectory_ids: np.ndarray
    parent_ids: np.ndarray

    @property
    def records(self) -> int:
        return int(self.indices.size)

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(self, field.name).nbytes) for field in fields(self))


@dataclass(frozen=True)
class TorchFeatureBatch:
    """Model inputs and targets; provenance remains in the NumPy batch."""

    line_patterns: Any
    own_gains: Any
    opponent_gains: Any
    teacher_values: Any
    two_ply_closure_values: Any
    residuals: Any
    plys: Any
    completed_depths: Any
    deepest_completed_depths: Any
    terminal_depths: Any
    flags: Any
    duplicate_counts: Any

    @property
    def records(self) -> int:
        return int(self.residuals.numel())

    @property
    def nbytes(self) -> int:
        return sum(
            int(getattr(self, field.name).numel() * getattr(self, field.name).element_size())
            for field in fields(self)
        )

    def to(self, device: str, *, non_blocking: bool = False) -> TorchFeatureBatch:
        return TorchFeatureBatch(**{
            field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
            for field in fields(self)
        })


class MappedFeatureDataset:
    """Authenticated fixed records backed directly by ``features.bin``."""

    def __init__(self, directory: Path | str, *, verify_digest: bool = True) -> None:
        self.artifact: FeatureArtifact = open_feature_artifact(
            directory, verify_digest=verify_digest, require_clean=True
        )
        self.records = np.memmap(
            self.artifact.binary_path,
            dtype=RECORD_DTYPE,
            mode="r",
            offset=HEADER.size,
            shape=(self.artifact.record_count,),
        )
        self._validate_columns()

    def _validate_columns(self) -> None:
        split = np.asarray(self.records["split"])
        counts = np.bincount(split, minlength=4)
        if tuple(int(value) for value in counts[1:4]) != self.artifact.split_counts:
            raise ValueError("record split counts differ from the manifest")
        if int(counts[0]) != 0 or int(counts[4:].sum()) != 0:
            raise ValueError("records contain an unknown split")

        teacher = np.asarray(self.records["teacher_value"], dtype=np.int64)
        closure_value = np.asarray(self.records["two_ply_closure_value"], dtype=np.int64)
        residual = np.asarray(self.records["residual"], dtype=np.int64)
        if not np.array_equal(residual, teacher - closure_value):
            raise ValueError("record residuals differ from teacher minus two-ply closure")

        key_low = np.asarray(self.records["key_low"])
        key_high = np.asarray(self.records["key_high"])
        out_of_order = ((key_low[1:] < key_low[:-1]) |
                        ((key_low[1:] == key_low[:-1]) &
                         (key_high[1:] <= key_high[:-1])))
        if bool(out_of_order.any()):
            raise ValueError("canonical keys are not uniquely sorted")

        patterns = np.asarray(self.records["line_patterns"])
        if int(patterns.max(initial=0)) >= 3276:
            raise ValueError("record contains an unknown line pattern")
        if bool(np.asarray(self.records["reserved"]).any()) or bool(
                np.asarray(self.records["trailing_reserved"]).any()):
            raise ValueError("record contains nonzero reserved data")

    def indices(self, split: str | None = None) -> np.ndarray:
        if split is None:
            return np.arange(self.artifact.record_count, dtype=np.int64)
        try:
            split_id = SPLITS[split]
        except KeyError as error:
            raise ValueError(f"unknown split: {split}") from error
        return np.flatnonzero(self.records["split"] == split_id).astype(np.int64, copy=False)

    def materialize(self, split: str | None = None) -> NumpyFeatureBatch:
        """Copy selected columns once; no Python record loop is involved."""
        indices = self.indices(split)
        selected = self.records if split is None else self.records[indices]

        def array(name: str, dtype: Any) -> np.ndarray:
            return np.ascontiguousarray(selected[name], dtype=dtype)

        return NumpyFeatureBatch(
            indices=np.ascontiguousarray(indices),
            line_patterns=array("line_patterns", np.int64),
            own_gains=array("own_gains", np.int16),
            opponent_gains=array("opponent_gains", np.int16),
            teacher_values=array("teacher_value", np.float32),
            two_ply_closure_values=array("two_ply_closure_value", np.float32),
            residuals=array("residual", np.float32),
            plys=array("ply", np.uint8),
            completed_depths=array("completed_depth", np.uint8),
            deepest_completed_depths=array("deepest_completed_depth", np.uint8),
            terminal_depths=array("terminal_depth", np.uint8),
            flags=array("flags", np.uint8),
            duplicate_counts=array("duplicate_count", np.int64),
            family_ids=array("family_id", np.uint64),
            trajectory_ids=array("trajectory_id", np.uint64),
            parent_ids=array("parent_id", np.uint64),
        )

    def close(self) -> None:
        mapping = getattr(self.records, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def __enter__(self) -> MappedFeatureDataset:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def to_torch(batch: NumpyFeatureBatch) -> TorchFeatureBatch:
    """Create zero-copy CPU tensors for the model-relevant contiguous arrays."""
    import torch

    names = (
        "line_patterns",
        "own_gains",
        "opponent_gains",
        "teacher_values",
        "two_ply_closure_values",
        "residuals",
        "plys",
        "completed_depths",
        "deepest_completed_depths",
        "terminal_depths",
        "flags",
        "duplicate_counts",
    )
    return TorchFeatureBatch(**{
        name: torch.from_numpy(getattr(batch, name))
        for name in names
    })


def subset_batch(batch: NumpyFeatureBatch, selected: np.ndarray) -> NumpyFeatureBatch:
    """Select aligned records from an already materialized NumPy batch."""
    mask = np.asarray(selected)
    if mask.shape != (batch.records,) or mask.dtype != np.bool_:
        raise ValueError("batch selection must be a boolean vector matching the record count")
    return NumpyFeatureBatch(**{
        field.name: np.ascontiguousarray(getattr(batch, field.name)[mask])
        for field in fields(batch)
    })
