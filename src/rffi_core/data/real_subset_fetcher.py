"""Strict real-only dataset subset fetcher with streaming and low RAM usage."""

from __future__ import annotations

import gzip
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rffi_core.utils.errors import ConfigError, DataContractError, MemorySafetyError


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One real sample entry exposed by a remote manifest."""

    sample_id: str
    url: str
    bytes_size: int
    is_real: bool
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RealSubsetConfig:
    """Configuration for downloading a deterministic tiny real-data subset."""

    manifest_url: str
    output_dir: Path
    subset_percent: float = 1.0
    timeout_seconds: int = 30
    max_total_download_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        parsed = urlparse(self.manifest_url)
        if parsed.scheme not in {"http", "https"}:
            raise ConfigError("manifest_url must use http or https")
        if not 1.0 <= self.subset_percent <= 100.0:
            raise ConfigError("subset_percent must be within [1.0, 100.0]")
        if self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds must be > 0")
        if self.max_total_download_bytes <= 0:
            raise ConfigError("max_total_download_bytes must be > 0")


@dataclass(frozen=True, slots=True)
class DownloadedItem:
    """Metadata describing a downloaded real sample."""

    sample_id: str
    source_url: str
    local_path: str
    bytes_size: int


@dataclass(frozen=True, slots=True)
class RealSubsetResult:
    """Fetch outcome for auditing and reproducibility."""

    manifest_url: str
    subset_percent: float
    total_manifest_entries: int
    selected_entries: int
    total_downloaded_bytes: int
    items: tuple[DownloadedItem, ...]


def _open_text_stream(url: str, timeout_seconds: int) -> io.TextIOBase:
    request = Request(url=url, headers={"User-Agent": "rffi-open-set/0.1"})
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except HTTPError as exc:
        raise ConfigError(f"HTTP error while opening manifest: {exc}") from exc
    except URLError as exc:
        raise ConfigError(f"URL error while opening manifest: {exc}") from exc

    if url.endswith(".gz"):
        return io.TextIOWrapper(gzip.GzipFile(fileobj=response), encoding="utf-8")
    return io.TextIOWrapper(response, encoding="utf-8")


def _iter_manifest_entries(manifest_url: str, timeout_seconds: int) -> Iterator[ManifestEntry]:
    with _open_text_stream(manifest_url, timeout_seconds) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                raw_obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataContractError(
                    f"Manifest line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(raw_obj, dict):
                raise DataContractError(f"Manifest line {line_number} must be a JSON object")

            typed_obj = cast(dict[str, Any], raw_obj)

            sample_id = typed_obj.get("sample_id")
            sample_url = typed_obj.get("url")
            bytes_size = typed_obj.get("bytes")
            is_real = typed_obj.get("is_real")
            sha256 = typed_obj.get("sha256")

            if not isinstance(sample_id, str) or sample_id.strip() == "":
                raise DataContractError(f"Manifest line {line_number} has invalid sample_id")
            if not isinstance(sample_url, str) or sample_url.strip() == "":
                raise DataContractError(f"Manifest line {line_number} has invalid url")
            if not isinstance(bytes_size, int) or bytes_size <= 0:
                raise DataContractError(f"Manifest line {line_number} has invalid bytes")
            if not isinstance(is_real, bool):
                raise DataContractError(f"Manifest line {line_number} must contain boolean is_real")
            if is_real is not True:
                raise DataContractError(
                    "Synthetic or non-real entry detected in manifest; this project permits real data only"
                )
            if sha256 is not None and not isinstance(sha256, str):
                raise DataContractError(f"Manifest line {line_number} has invalid sha256")

            yield ManifestEntry(
                sample_id=sample_id,
                url=sample_url,
                bytes_size=bytes_size,
                is_real=is_real,
                sha256=sha256,
            )


def _count_entries(config: RealSubsetConfig) -> int:
    return sum(1 for _ in _iter_manifest_entries(config.manifest_url, config.timeout_seconds))


def _target_count(total_entries: int, subset_percent: float) -> int:
    desired = math.ceil(total_entries * (subset_percent / 100.0))
    if desired < 1:
        return 1
    if desired > total_entries:
        return total_entries
    return desired


def _should_pick(index: int, total_entries: int, selected_entries: int, picked_so_far: int) -> bool:
    current = (picked_so_far * total_entries) // selected_entries
    return index == current


def _download_to_file(source_url: str, destination: Path, timeout_seconds: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url=source_url, headers={"User-Agent": "rffi-open-set/0.1"})

    try:
        with urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as sink:
            total = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                sink.write(chunk)
                total += len(chunk)
            return total
    except HTTPError as exc:
        raise ConfigError(f"HTTP error while downloading sample: {source_url} | {exc}") from exc
    except URLError as exc:
        raise ConfigError(f"URL error while downloading sample: {source_url} | {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Filesystem error while writing sample: {destination} | {exc}") from exc


def fetch_real_subset(config: RealSubsetConfig) -> RealSubsetResult:
    """Fetch an exact percentage subset of real entries from a streaming manifest."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    total_entries = _count_entries(config)
    if total_entries == 0:
        raise DataContractError("Manifest has zero entries")

    selected_entries = _target_count(total_entries, config.subset_percent)
    total_downloaded = 0
    picked_so_far = 0
    downloaded: list[DownloadedItem] = []

    for index, entry in enumerate(
        _iter_manifest_entries(config.manifest_url, config.timeout_seconds),
        start=0,
    ):
        if picked_so_far >= selected_entries:
            break
        if not _should_pick(index, total_entries, selected_entries, picked_so_far):
            continue

        sample_ext = Path(urlparse(entry.url).path).suffix
        if sample_ext == "":
            sample_ext = ".bin"
        target = config.output_dir / f"{entry.sample_id}{sample_ext}"

        downloaded_bytes = _download_to_file(entry.url, target, config.timeout_seconds)
        total_downloaded += downloaded_bytes
        if total_downloaded > config.max_total_download_bytes:
            raise MemorySafetyError(
                "Downloaded bytes exceeded local safety budget. "
                f"Current: {total_downloaded}, budget: {config.max_total_download_bytes}"
            )

        downloaded.append(
            DownloadedItem(
                sample_id=entry.sample_id,
                source_url=entry.url,
                local_path=str(target),
                bytes_size=downloaded_bytes,
            )
        )
        picked_so_far += 1

    if picked_so_far != selected_entries:
        raise DataContractError(
            "Unable to select the expected number of entries from manifest. "
            f"Expected {selected_entries}, got {picked_so_far}"
        )

    result = RealSubsetResult(
        manifest_url=config.manifest_url,
        subset_percent=config.subset_percent,
        total_manifest_entries=total_entries,
        selected_entries=selected_entries,
        total_downloaded_bytes=total_downloaded,
        items=tuple(downloaded),
    )

    manifest_out = config.output_dir / "subset_manifest.json"
    try:
        manifest_out.write_text(
            json.dumps(
                {
                    "manifest_url": result.manifest_url,
                    "subset_percent": result.subset_percent,
                    "total_manifest_entries": result.total_manifest_entries,
                    "selected_entries": result.selected_entries,
                    "total_downloaded_bytes": result.total_downloaded_bytes,
                    "items": [
                        {
                            "sample_id": item.sample_id,
                            "source_url": item.source_url,
                            "local_path": item.local_path,
                            "bytes_size": item.bytes_size,
                        }
                        for item in result.items
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigError(f"Failed to write subset manifest: {exc}") from exc

    return result
