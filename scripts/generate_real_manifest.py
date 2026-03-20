"""Generate a strict real-only NDJSON manifest from a public Zenodo RF dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class DatasetChoice:
    """Public dataset descriptor used for manifest generation."""

    name: str
    description: str
    zenodo_record_id: int


DEFAULT_DATASET = DatasetChoice(
    name="rf_fingerprinting_wlan_2026",
    description=(
        "Zenodo record 18515187 (2026): RF Fingerprinting WLAN Dataset. "
        "Public real captured RF data with anechoic and office environments."
    ),
    zenodo_record_id=18515187,
)


def _fetch_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = Request(url=url, headers={"User-Agent": "rffi-open-set/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error while fetching {url}: {exc}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error while fetching {url}: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload from {url}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Zenodo record payload must be a JSON object")
    if not all(isinstance(key, str) for key in cast(dict[Any, Any], parsed)):
        raise RuntimeError("Zenodo record payload keys must be strings")
    return cast(dict[str, Any], parsed)


def _iter_manifest_rows(record_json: dict[str, Any]) -> list[dict[str, Any]]:
    files_obj = record_json.get("files")
    if not isinstance(files_obj, list):
        raise RuntimeError("Zenodo record has no downloadable files")
    files_list = cast(list[object], files_obj)
    if len(files_list) == 0:
        raise RuntimeError("Zenodo record has no downloadable files")

    rows: list[dict[str, Any]] = []
    for index, file_obj in enumerate(files_list, start=1):
        if not isinstance(file_obj, dict):
            raise RuntimeError(f"File entry {index} is malformed")
        typed_file = cast(dict[str, Any], file_obj)

        key = typed_file.get("key")
        size = typed_file.get("size")
        links_obj = typed_file.get("links")

        if not isinstance(key, str) or key.strip() == "":
            raise RuntimeError(f"File entry {index} missing valid key")
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"File entry {index} missing valid size")
        if not isinstance(links_obj, dict):
            raise RuntimeError(f"File entry {index} missing links object")
        typed_links = cast(dict[str, Any], links_obj)

        url = typed_links.get("self")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise RuntimeError(f"File entry {index} missing valid downloadable URL")

        sample_id = Path(key).stem
        rows.append(
            {
                "sample_id": sample_id,
                "url": url,
                "bytes": size,
                "is_real": True,
            }
        )

    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate .ndjson manifest with real RF dataset URLs and byte sizes"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET.name,
        choices=[DEFAULT_DATASET.name],
        help="Dataset preset to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/real_dataset_manifest.ndjson"),
        help="Output NDJSON manifest path",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Network timeout in seconds",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = DEFAULT_DATASET
    if args.dataset != dataset.name:
        print("Unsupported dataset choice", file=sys.stderr)
        return 2

    if args.timeout_seconds <= 0:
        print("--timeout-seconds must be > 0", file=sys.stderr)
        return 2

    record_url = f"https://zenodo.org/api/records/{dataset.zenodo_record_id}"
    try:
        record_json = _fetch_json(record_url, args.timeout_seconds)
        rows = _iter_manifest_rows(record_json)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, separators=(",", ":")))
                sink.write("\n")
    except OSError as exc:
        print(f"ERROR: failed writing manifest {args.output}: {exc}", file=sys.stderr)
        return 2

    print("Generated real-only NDJSON manifest")
    print(f"Dataset: {dataset.name}")
    print(f"Description: {dataset.description}")
    print(f"Record: {record_url}")
    print(f"Entries: {len(rows)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
