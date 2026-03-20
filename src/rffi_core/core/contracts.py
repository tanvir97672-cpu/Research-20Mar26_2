"""Typed contracts for experiment protocol integrity."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SplitContract:
    """Dataset split policy contract for open-set experiments."""

    train_legit_ids: tuple[str, ...] = field(default_factory=tuple)
    val_legit_ids: tuple[str, ...] = field(default_factory=tuple)
    test_legit_ids: tuple[str, ...] = field(default_factory=tuple)
    test_rogue_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.train_legit_ids) == 0:
            raise ValueError("train_legit_ids must not be empty")
        if len(self.test_rogue_ids) == 0:
            raise ValueError("test_rogue_ids must not be empty")


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    """Top-level immutable research protocol contract."""

    protocol_name: str
    split: SplitContract
    real_data_only: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.protocol_name.strip() == "":
            raise ValueError("protocol_name must not be empty")
        if not self.real_data_only:
            raise ValueError("real_data_only must remain True for this project")
