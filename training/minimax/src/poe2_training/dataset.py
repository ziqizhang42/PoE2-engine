"""Vectorized, memory-mapped access to minimax feature records."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import HEADER, RECORD_BYTES, FeatureArtifact, open_feature_artifact


SPLITS = {"train": 1, "validation": 2, "test": 3}
KEY_DTYPE = np.dtype([("low", "<u8"), ("high", "<u8")])
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
        return _batch(indices, self.records if split is None else self.records[indices])

    def select(self, indices: np.ndarray) -> NumpyFeatureBatch:
        """Copy the columns of the given ascending record indices."""
        return _batch(indices, self.records[indices])

    def close(self) -> None:
        mapping = getattr(self.records, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def __enter__(self) -> MappedFeatureDataset:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _batch(indices: np.ndarray, selected: np.ndarray) -> NumpyFeatureBatch:
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


def canonical_keys(records: np.ndarray) -> np.ndarray:
    """Return D4-canonical position keys in the (low, high) order features are sorted by."""
    keys = np.empty(records.shape[0], dtype=KEY_DTYPE)
    keys["low"] = records["key_low"]
    keys["high"] = records["key_high"]
    return keys


def key_membership(keys: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Return membership in a sorted unique canonical-key array."""
    candidates = np.asarray(keys, dtype=KEY_DTYPE)
    reference = np.asarray(known, dtype=KEY_DTYPE)
    if reference.size == 0:
        return np.zeros(candidates.size, dtype=np.bool_)
    offsets = np.searchsorted(reference, candidates)
    in_range = offsets < reference.size
    result = np.zeros(candidates.size, dtype=np.bool_)
    result[in_range] = reference[offsets[in_range]] == candidates[in_range]
    return result


def _family_order(family_ids: np.ndarray) -> np.ndarray:
    """Return a deterministic order of families that does not follow their ID order.

    SplitMix64 spreads consecutive family IDs, so a capped subset is not skewed toward the
    trajectories generated first.
    """
    with np.errstate(over="ignore"):
        mixed = family_ids + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
    return np.lexsort((family_ids, mixed))


@dataclass(frozen=True)
class TrainingPart:
    """One dataset's train records after the leak, duplicate, and cap rules."""

    dataset: MappedFeatureDataset
    weight: float
    indices: np.ndarray
    train_records: int
    dropped_primary_holdout: int
    dropped_newer_duplicate: int
    dropped_cap: int

    def describe(self) -> dict[str, Any]:
        artifact = self.dataset.artifact
        model = artifact.manifest.get("inputs", {}).get("label_search", {}).get(
            "model_header_sha256")
        return {
            "corpus_id": artifact.manifest["corpus"]["id"],
            "feature_binary_sha256": artifact.binary_digest,
            "feature_manifest_sha256": artifact.manifest_digest,
            "teacher": None if model is None else f"candidate-sha256-{model}",
            "weight": self.weight,
            "train_records": self.train_records,
            "dropped_primary_holdout": self.dropped_primary_holdout,
            "dropped_newer_duplicate": self.dropped_newer_duplicate,
            "dropped_cap": self.dropped_cap,
            "records_used": int(self.indices.size),
        }


class TrainingSelection:
    """The train records of a primary dataset and weighted extra datasets, newest first.

    The one leak rule: extra train records whose position is in the primary's validation or
    test split are dropped, so those splits stay unseen. A position present in several
    datasets is kept only in the first (newest) one, and an optional cap keeps whole datasets
    in order, then whole games of the first dataset that does not fit, then nothing more.
    """

    def __init__(self, primary: MappedFeatureDataset,
                 extras: Sequence[tuple[MappedFeatureDataset, float]] = (),
                 *, max_records: int | None = None) -> None:
        primary_keys = canonical_keys(primary.records)
        held_out = primary_keys[np.asarray(primary.records["split"]) != SPLITS["train"]]
        taken = np.empty(0, dtype=KEY_DTYPE)
        remaining = max_records
        parts: list[TrainingPart] = []
        for dataset, weight in ((primary, 1.0), *extras):
            if not (math.isfinite(weight) and weight > 0.0):
                raise ValueError("training dataset weights must be positive and finite")
            indices = dataset.indices("train")
            keys = canonical_keys(dataset.records[indices])
            leaked = key_membership(keys, held_out)
            repeated = key_membership(keys, taken) & ~leaked
            eligible = indices[~(leaked | repeated)]
            selected = self._capped(dataset, eligible, remaining)
            if remaining is not None:
                remaining = 0 if selected.size < eligible.size else remaining - int(selected.size)
            parts.append(TrainingPart(
                dataset=dataset, weight=float(weight), indices=selected,
                train_records=int(indices.size), dropped_primary_holdout=int(leaked.sum()),
                dropped_newer_duplicate=int(repeated.sum()),
                dropped_cap=int(eligible.size - selected.size)))
            taken = np.union1d(taken, canonical_keys(dataset.records[selected]))
        self.parts = tuple(parts)
        self.max_records = max_records
        if self.records == 0:
            raise ValueError("training selection is empty")

    @staticmethod
    def _capped(dataset: MappedFeatureDataset, eligible: np.ndarray,
                remaining: int | None) -> np.ndarray:
        if remaining is None or eligible.size <= remaining:
            return eligible
        if remaining <= 0:
            return eligible[:0]
        families = np.asarray(dataset.records["family_id"][eligible], dtype=np.uint64)
        unique, sizes = np.unique(families, return_counts=True)
        order = _family_order(unique)
        fitting = np.cumsum(sizes[order]) <= remaining
        chosen = unique[order][:int(np.argmin(fitting)) if not fitting.all() else fitting.size]
        return eligible[np.isin(families, chosen)]

    @property
    def records(self) -> int:
        return sum(int(part.indices.size) for part in self.parts)

    @property
    def uniform(self) -> bool:
        """Whether every used record has the same weight, so the plain mean applies."""
        return len({part.weight for part in self.parts if part.indices.size}) <= 1

    def pieces(self, piece_records: int) -> Iterator[tuple[NumpyFeatureBatch, np.ndarray]]:
        """Yield at most piece_records records at a time with their weights, in dataset order."""
        if piece_records < 1:
            raise ValueError("piece_records must be positive")
        for part in self.parts:
            for start in range(0, part.indices.size, piece_records):
                indices = part.indices[start:start + piece_records]
                yield part.dataset.select(indices), np.full(indices.size, part.weight)

    def materialize(self) -> tuple[NumpyFeatureBatch, np.ndarray]:
        """Return every selected record as one batch with aligned weights."""
        batches = [(part.dataset.select(part.indices), np.full(part.indices.size, part.weight))
                   for part in self.parts]
        return (NumpyFeatureBatch(**{
            field.name: np.ascontiguousarray(np.concatenate(
                [getattr(batch, field.name) for batch, _ in batches]))
            for field in fields(NumpyFeatureBatch)
        }), np.concatenate([weights for _, weights in batches]))

    def describe(self) -> list[dict[str, Any]]:
        return [part.describe() for part in self.parts]
