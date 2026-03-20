"""Typed configuration schemas with strict stdlib validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Filesystem locations used by the project."""

    project_root: Path = Path(".")
    metadata_dir: Path = Path("metadata")
    raw_data_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("artifacts")

    def __post_init__(self) -> None:
        for value in (
            self.project_root,
            self.metadata_dir,
            self.raw_data_dir,
            self.artifacts_dir,
        ):
            if str(value).strip() == "":
                raise ValueError("Path values must not be empty")


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    """Conservative runtime limits for low-memory local development."""

    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    local_batch_upper_bound: int = 16
    cloud_batch_upper_bound: int = 64

    def __post_init__(self) -> None:
        if not 0 <= self.num_workers <= 2:
            raise ValueError("num_workers must be in [0, 2]")
        if not 1 <= self.prefetch_factor <= 4:
            raise ValueError("prefetch_factor must be in [1, 4]")
        if not 1 <= self.local_batch_upper_bound <= 64:
            raise ValueError("local_batch_upper_bound must be in [1, 64]")
        if not 1 <= self.cloud_batch_upper_bound <= 256:
            raise ValueError("cloud_batch_upper_bound must be in [1, 256]")
        if self.cloud_batch_upper_bound < self.local_batch_upper_bound:
            raise ValueError("cloud_batch_upper_bound must be >= local_batch_upper_bound")
