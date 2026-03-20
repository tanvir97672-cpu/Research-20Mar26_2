"""Shared type aliases and protocols."""

from __future__ import annotations

from pathlib import Path
from typing import NewType, TypeAlias

DeviceId = NewType("DeviceId", str)
SessionId = NewType("SessionId", str)

PathLike: TypeAlias = str | Path
