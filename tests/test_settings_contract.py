"""Contract-level tests for settings validation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from rffi_core.config.settings import load_settings
from rffi_core.utils.errors import ConfigError


def test_load_settings_raises_for_missing_file() -> None:
    missing = Path("configs/does-not-exist.yaml")
    with pytest.raises(ConfigError):
        _ = load_settings(missing)
