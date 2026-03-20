"""CLI for fetching an exact tiny real-data subset from a remote manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rffi_core.data.real_subset_fetcher import RealSubsetConfig, fetch_real_subset
from rffi_core.utils.errors import ProjectError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download an exact subset percentage of real-only data entries from a remote NDJSON manifest."
        )
    )
    parser.add_argument(
        "--manifest-url",
        required=True,
        type=str,
        help="HTTPS/HTTP URL to an NDJSON manifest (or .ndjson.gz) of real samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/subset_1pct"),
        help="Directory where the subset and local subset manifest will be written",
    )
    parser.add_argument(
        "--subset-percent",
        type=float,
        default=1.0,
        help="Subset percentage in [1.0, 100.0]. Default is 1.0",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Network timeout per request in seconds",
    )
    parser.add_argument(
        "--max-total-download-mib",
        type=int,
        default=512,
        help="Safety budget to prevent local memory/storage pressure",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        cfg = RealSubsetConfig(
            manifest_url=args.manifest_url,
            output_dir=args.output_dir,
            subset_percent=args.subset_percent,
            timeout_seconds=args.timeout_seconds,
            max_total_download_bytes=args.max_total_download_mib * 1024 * 1024,
        )
        result = fetch_real_subset(cfg)
    except ProjectError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Real subset fetch completed")
    print(f"Manifest URL: {result.manifest_url}")
    print(f"Subset percent: {result.subset_percent}")
    print(f"Total entries in manifest: {result.total_manifest_entries}")
    print(f"Selected entries: {result.selected_entries}")
    print(f"Total downloaded bytes: {result.total_downloaded_bytes}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
