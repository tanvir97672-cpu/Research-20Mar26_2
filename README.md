# rffi-open-set

Foundational Python codebase for real-only open-set RFFI research.

## Principles

- Real-data only: no synthetic generation, no mock sample creation.
- Safe local development on low-memory hardware.
- Strict static typing and defensive validation to reduce cloud runtime failures.
- No training/data execution logic included in this foundation.

## Environment

- Local dev target: CPU-only constrained laptop.
- Cloud target: Lightning AI with L4 GPU.
- Python: 3.11.

## Install

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Optional GPU runtime dependencies can be installed only on cloud:

```bash
pip install -e ".[ml]"
```

## Static checks

```bash
ruff check .
mypy src tests
pyright
pytest
```

## Project layout

- `src/rffi_core/config`: strict settings and schema models.
- `src/rffi_core/core`: core contracts and safety guards.
- `src/rffi_core/utils`: shared utilities and error types.
- `tests`: placeholder for future tests.

## Notes

- Keep configuration in files or environment variables; never hardcode machine-specific paths.
- Use `configs/default.json` as the strict configuration template.
- Add execution modules later under explicit entry points once data and protocols are finalized.

## Real-only tiny subset fetch

Use the script below to stream a remote manifest and download exactly 1% (or another value >= 1%) of real samples only.

First, generate a manifest from the public real RF dataset preset:

```bash
python scripts/generate_real_manifest.py \
	--dataset rf_fingerprinting_wlan_2026 \
	--output configs/real_dataset_manifest.ndjson
```

```bash
python scripts/fetch_real_subset.py \
	--manifest-url "https://your-dataset-host/path/manifest.ndjson.gz" \
	--output-dir data/raw/subset_1pct \
	--subset-percent 1.0 \
	--max-total-download-mib 512
```

Manifest format is documented in `configs/real_manifest_schema.md`.

Minimal end-to-end smoke training loop on downloaded real subset:

```bash
python scripts/run_smoke_test.py --device cpu --dry-run
```

L4-optimized smoke command (Lightning GPU environment):

```bash
python scripts/run_smoke_test.py --device cuda --profile l4 --feature-dim 256 --epochs 2 --batch-size 256 --max-train-steps 200
```

Tested local flow (serves generated manifest over HTTP so fetcher can stream it):

```bash
python -m http.server 8765
python scripts/fetch_real_subset.py --manifest-url "http://127.0.0.1:8765/configs/real_dataset_manifest.ndjson" --output-dir data/raw/subset_1pct --subset-percent 1.0 --max-total-download-mib 512
python scripts/run_smoke_test.py --device cpu --dry-run
```
