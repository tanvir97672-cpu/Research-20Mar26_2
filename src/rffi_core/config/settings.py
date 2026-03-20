"""Application settings with strict stdlib-only parsing and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from typing import cast

from rffi_core.config.schemas import DataPaths, RuntimeLimits
from rffi_core.utils.errors import ConfigError


@dataclass(frozen=True, slots=True)
class LightningConfig:
    """Cloud runtime details for Lightning AI target environment."""

    accelerator: str = "gpu"
    gpu_type: str = "L4"
    mixed_precision: str = "16-mixed"


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Top-level immutable application settings."""

    project_name: str = "rffi-open-set"
    debug: bool = False
    data: DataPaths = DataPaths()
    runtime: RuntimeLimits = RuntimeLimits()
    lightning: LightningConfig = LightningConfig()


@dataclass(frozen=True, slots=True)
class _RawSettings:
    """Intermediate parsed settings from file before coercion."""

    project_name: str | None = None
    debug: bool | None = None
    data: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    lightning: Mapping[str, Any] | None = None


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{field_name}' must be a mapping/object")
    typed: dict[str, Any] = {}
    for key, item in cast(Mapping[Any, Any], value).items():
        if not isinstance(key, str):
            raise ConfigError(f"'{field_name}' keys must all be strings")
        typed[key] = item
    return typed


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Environment variable {name} must be a boolean-like value")


def _parse_root(payload: Mapping[str, Any]) -> _RawSettings:
    allowed = {"project_name", "debug", "data", "runtime", "lightning"}
    unknown = set(payload.keys()) - allowed
    if unknown:
        raise ConfigError(f"Unknown top-level settings keys: {sorted(unknown)}")

    return _RawSettings(
        project_name=payload.get("project_name") if isinstance(payload.get("project_name"), str) else None,
        debug=payload.get("debug") if isinstance(payload.get("debug"), bool) else None,
        data=_as_mapping(payload["data"], "data") if "data" in payload else None,
        runtime=_as_mapping(payload["runtime"], "runtime") if "runtime" in payload else None,
        lightning=_as_mapping(payload["lightning"], "lightning") if "lightning" in payload else None,
    )


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load settings from optional JSON file, then optional env overrides."""
    base = AppSettings()

    if config_path is not None:
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        try:
            content = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Malformed JSON configuration: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"Unable to read configuration file: {exc}") from exc

        if not isinstance(content, dict):
            raise ConfigError("Configuration root must be a JSON object")
        if not all(isinstance(key, str) for key in cast(dict[Any, Any], content)):
            raise ConfigError("Configuration root keys must all be strings")

        parsed = _parse_root(cast(dict[str, Any], content))
        data = DataPaths(**dict(parsed.data)) if parsed.data is not None else base.data
        runtime = RuntimeLimits(**dict(parsed.runtime)) if parsed.runtime is not None else base.runtime
        lightning = (
            LightningConfig(**dict(parsed.lightning))
            if parsed.lightning is not None
            else base.lightning
        )
        base = AppSettings(
            project_name=parsed.project_name or base.project_name,
            debug=base.debug if parsed.debug is None else parsed.debug,
            data=data,
            runtime=runtime,
            lightning=lightning,
        )

    env_debug = _parse_bool_env("RFFI_DEBUG", base.debug)
    env_project_name = os.getenv("RFFI_PROJECT_NAME", base.project_name)

    if env_project_name.strip() == "":
        raise ConfigError("RFFI_PROJECT_NAME must not be empty when provided")

    return AppSettings(
        project_name=env_project_name,
        debug=env_debug,
        data=base.data,
        runtime=base.runtime,
        lightning=base.lightning,
    )
