"""Project-specific exception hierarchy for explicit, typed error handling."""

from __future__ import annotations


class ProjectError(Exception):
    """Base exception for all project-level errors."""


class ConfigError(ProjectError):
    """Raised when configuration is invalid or incomplete."""


class DataContractError(ProjectError):
    """Raised when data metadata/contracts are violated."""


class MemorySafetyError(ProjectError):
    """Raised when requested runtime settings exceed safe limits."""
