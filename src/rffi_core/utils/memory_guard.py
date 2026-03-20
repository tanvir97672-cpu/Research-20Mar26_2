"""Memory safety heuristics for constrained local development."""

from __future__ import annotations

from dataclasses import dataclass

from rffi_core.utils.errors import MemorySafetyError


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """Simple memory budget model in MiB for conservative local checks."""

    max_process_ram_mib: int = 6144
    max_batch_ram_mib: int = 1024


def validate_batch_budget(
    *,
    estimated_batch_ram_mib: int,
    budget: MemoryBudget | None = None,
) -> None:
    """Validate a proposed batch memory estimate against budget constraints."""
    active_budget = budget or MemoryBudget()
    if estimated_batch_ram_mib <= 0:
        raise MemorySafetyError("estimated_batch_ram_mib must be a positive integer")
    if estimated_batch_ram_mib > active_budget.max_batch_ram_mib:
        raise MemorySafetyError(
            "Estimated batch RAM exceeds safe local development threshold: "
            f"{estimated_batch_ram_mib} MiB > {active_budget.max_batch_ram_mib} MiB"
        )
