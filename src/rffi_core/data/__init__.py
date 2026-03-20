"""Data access package for strict real-only dataset workflows."""

from rffi_core.data.real_subset_fetcher import (
    RealSubsetConfig,
    RealSubsetResult,
    fetch_real_subset,
)

__all__ = ["RealSubsetConfig", "RealSubsetResult", "fetch_real_subset"]
