"""Minimal deterministic smoke training loop using downloaded real subset data only."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for run_smoke_test.py. Install optional dependency group [ml]."
    ) from exc


@dataclass(frozen=True, slots=True)
class LocalItem:
    """One locally downloaded real sample item."""

    sample_id: str
    local_path: Path
    class_index: int


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    """Runtime settings for minimal forward/backward smoke test."""

    subset_manifest_path: Path
    feature_dim: int
    epochs: int
    max_files: int
    max_entries_per_zip: int
    bytes_per_entry: int
    batch_size: int
    max_train_steps: int
    device: str
    profile: str
    report_json: Path | None
    dry_run: bool


class DeterministicSmokeModel(nn.Module):
    """Tiny deterministic model with reconstruction and classification heads."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        self.recon_weight = nn.Parameter(torch.eye(feature_dim, dtype=torch.float32), requires_grad=True)
        self.recon_bias = nn.Parameter(torch.zeros(feature_dim, dtype=torch.float32), requires_grad=True)
        self.cls_weight = nn.Parameter(torch.zeros(num_classes, feature_dim, dtype=torch.float32), requires_grad=True)
        self.cls_bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32), requires_grad=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        recon = x @ self.recon_weight.T + self.recon_bias
        logits = x @ self.cls_weight.T + self.cls_bias
        return recon, logits


def _load_subset_manifest(path: Path) -> tuple[list[LocalItem], list[str]]:
    if not path.exists():
        raise RuntimeError(f"subset manifest does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read subset manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"subset manifest is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("subset manifest root must be a JSON object")
    if not all(isinstance(key, str) for key in cast(dict[Any, Any], payload)):
        raise RuntimeError("subset manifest root keys must be strings")

    payload_obj = cast(dict[str, Any], payload)

    items_obj = payload_obj.get("items")
    if not isinstance(items_obj, list):
        raise RuntimeError("subset manifest contains no items")

    items_list = cast(list[object], items_obj)
    if len(items_list) == 0:
        raise RuntimeError("subset manifest contains no items")

    raw_items: list[tuple[str, Path]] = []
    for idx, raw in enumerate(items_list, start=1):
        if not isinstance(raw, dict):
            raise RuntimeError(f"subset manifest item {idx} is malformed")
        typed_raw = cast(dict[str, Any], raw)

        sample_id = typed_raw.get("sample_id")
        local_path = typed_raw.get("local_path")
        if not isinstance(sample_id, str) or sample_id.strip() == "":
            raise RuntimeError(f"subset manifest item {idx} has invalid sample_id")
        if not isinstance(local_path, str) or local_path.strip() == "":
            raise RuntimeError(f"subset manifest item {idx} has invalid local_path")

        file_path = Path(local_path)
        if not file_path.exists():
            raise RuntimeError(f"subset data file does not exist: {file_path}")

        raw_items.append((sample_id, file_path))

    class_names = sorted({sample_id for sample_id, _ in raw_items})
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    items: list[LocalItem] = []
    for sample_id, file_path in raw_items:
        items.append(
            LocalItem(
                sample_id=sample_id,
                local_path=file_path,
                class_index=class_to_index[sample_id],
            )
        )

    return items, class_names


def _bytes_to_pair(raw: bytes, feature_dim: int) -> tuple[Tensor, Tensor] | None:
    required = feature_dim * 2
    if len(raw) < required:
        return None

    x_raw = raw[:feature_dim]
    y_raw = raw[feature_dim:required]

    x = torch.tensor(list(x_raw), dtype=torch.float32) / 255.0
    y = torch.tensor(list(y_raw), dtype=torch.float32) / 255.0
    return x, y


def _iter_real_pairs(items: list[LocalItem], cfg: SmokeConfig) -> Iterator[tuple[Tensor, Tensor, int]]:
    yielded = 0
    for item in items[: cfg.max_files]:
        if yielded >= cfg.max_files:
            break

        suffix = item.local_path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(item.local_path, mode="r") as archive:
                inner_count = 0
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    if inner_count >= cfg.max_entries_per_zip:
                        break
                    with archive.open(info, mode="r") as reader:
                        raw = reader.read(cfg.bytes_per_entry)
                    pair = _bytes_to_pair(raw, cfg.feature_dim)
                    if pair is None:
                        continue
                    inner_count += 1
                    yielded += 1
                    x, y = pair
                    yield x, y, item.class_index
                    if yielded >= cfg.max_files:
                        break
        else:
            raw = item.local_path.read_bytes()[: cfg.bytes_per_entry]
            pair = _bytes_to_pair(raw, cfg.feature_dim)
            if pair is None:
                continue
            yielded += 1
            x, y = pair
            yield x, y, item.class_index


def _iter_real_batches(items: list[LocalItem], cfg: SmokeConfig) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    x_buf: list[Tensor] = []
    y_buf: list[Tensor] = []
    t_buf: list[int] = []
    for x, y, label in _iter_real_pairs(items, cfg):
        x_buf.append(x)
        y_buf.append(y)
        t_buf.append(label)
        if len(x_buf) >= cfg.batch_size:
            x_batch = torch.stack(x_buf, dim=0)
            y_batch = torch.stack(y_buf, dim=0)
            t_batch = torch.tensor(t_buf, dtype=torch.long)
            yield x_batch, y_batch, t_batch
            x_buf.clear()
            y_buf.clear()
            t_buf.clear()

    if len(x_buf) > 0:
        x_batch = torch.stack(x_buf, dim=0)
        y_batch = torch.stack(y_buf, dim=0)
        t_batch = torch.tensor(t_buf, dtype=torch.long)
        yield x_batch, y_batch, t_batch


def _compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> dict[str, Any]:
    if num_classes <= 0:
        raise ValueError("num_classes must be > 0")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths must match")

    matrix: list[list[int]] = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        if not (0 <= truth < num_classes) or not (0 <= pred < num_classes):
            raise ValueError("class index out of bounds")
        matrix[truth][pred] += 1

    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(num_classes))
    accuracy = (correct / total) if total > 0 else 0.0

    per_class: list[dict[str, Any]] = []
    precision_macro_sum = 0.0
    recall_macro_sum = 0.0
    f1_macro_sum = 0.0

    for cls in range(num_classes):
        tp = matrix[cls][cls]
        fp = sum(matrix[row][cls] for row in range(num_classes) if row != cls)
        fn = sum(matrix[cls][col] for col in range(num_classes) if col != cls)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_macro_sum += precision
        recall_macro_sum += recall
        f1_macro_sum += f1

        per_class.append(
            {
                "class_index": cls,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    denom = float(num_classes)
    precision_macro = precision_macro_sum / denom
    recall_macro = recall_macro_sum / denom
    f1_macro = f1_macro_sum / denom

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal deterministic forward/backward smoke loop on 1% real data subset"
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=Path("data/raw/subset_1pct/subset_manifest.json"),
        help="Path to subset_manifest.json produced by fetch_real_subset.py",
    )
    parser.add_argument("--feature-dim", type=int, default=256, help="Input/target vector size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of smoke epochs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--profile",
        type=str,
        default="laptop",
        choices=["laptop", "l4"],
        help="Execution profile: laptop-safe or L4-optimized",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size override; if 0, profile-based default is used",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Max optimizer steps override; if 0, profile-based default is used",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use minimal safe limits for potato-laptop mechanics testing",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write a structured JSON report with all metrics",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.feature_dim <= 0:
        print("ERROR: --feature-dim must be > 0", file=sys.stderr)
        return 2
    if args.epochs <= 0:
        print("ERROR: --epochs must be > 0", file=sys.stderr)
        return 2
    if args.batch_size < 0:
        print("ERROR: --batch-size must be >= 0", file=sys.stderr)
        return 2
    if args.max_train_steps < 0:
        print("ERROR: --max-train-steps must be >= 0", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available", file=sys.stderr)
        return 2

    if args.profile == "l4" and args.device != "cuda":
        print("ERROR: profile l4 requires --device cuda", file=sys.stderr)
        return 2

    if args.dry_run:
        default_max_files = 2
        default_entries_per_zip = 1
        default_batch_size = 1
        default_steps = 1
    elif args.profile == "l4":
        default_max_files = 64
        default_entries_per_zip = 16
        default_batch_size = 256
        default_steps = 200
    else:
        default_max_files = 16
        default_entries_per_zip = 4
        default_batch_size = 8
        default_steps = 25

    resolved_batch_size = args.batch_size if args.batch_size > 0 else default_batch_size
    resolved_steps = args.max_train_steps if args.max_train_steps > 0 else default_steps

    cfg = SmokeConfig(
        subset_manifest_path=args.subset_manifest,
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        max_files=default_max_files,
        max_entries_per_zip=default_entries_per_zip,
        bytes_per_entry=max(args.feature_dim * 2, 4096),
        batch_size=resolved_batch_size,
        max_train_steps=resolved_steps,
        device=args.device,
        profile=args.profile,
        report_json=args.report_json,
        dry_run=args.dry_run,
    )

    try:
        items, class_names = _load_subset_manifest(cfg.subset_manifest_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    model = DeterministicSmokeModel(cfg.feature_dim, len(class_names)).to(cfg.device)
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    recon_criterion = nn.MSELoss(reduction="mean")
    cls_criterion = nn.CrossEntropyLoss(reduction="mean")
    use_amp = cfg.device == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    total_steps = 0
    total_samples = 0
    last_loss = 0.0
    last_recon_loss = 0.0
    last_cls_loss = 0.0
    start_time = time.perf_counter()
    y_true: list[int] = []
    y_pred: list[int] = []

    model.train()
    for _ in range(cfg.epochs):
        for x_cpu, y_cpu, t_cpu in _iter_real_batches(items, cfg):
            x = x_cpu.to(cfg.device, non_blocking=use_amp)
            y = y_cpu.to(cfg.device, non_blocking=use_amp)
            t = t_cpu.to(cfg.device, non_blocking=use_amp)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                recon, logits = model(x)
                recon_loss = recon_criterion(recon, y)
                cls_loss = cls_criterion(logits, t)
                loss = recon_loss + cls_loss
            scaler.scale(loss).backward()
            scaler.step(cast(Any, optimizer))
            scaler.update()

            pred = torch.argmax(logits.detach(), dim=1)
            y_true.extend(t.detach().cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

            total_steps += 1
            total_samples += int(x.shape[0])
            last_loss = float(loss.detach().cpu().item())
            last_recon_loss = float(recon_loss.detach().cpu().item())
            last_cls_loss = float(cls_loss.detach().cpu().item())

            if total_steps >= cfg.max_train_steps:
                break
        if total_steps >= cfg.max_train_steps:
            break

    if total_steps == 0:
        print("ERROR: no valid real training pairs could be extracted from downloaded subset", file=sys.stderr)
        return 2

    elapsed = time.perf_counter() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0
    metrics = _compute_classification_metrics(y_true, y_pred, len(class_names))

    print("Smoke test completed")
    print(f"Device: {cfg.device}")
    print(f"Profile: {cfg.profile}")
    print(f"Dry run: {cfg.dry_run}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Steps: {total_steps}")
    print(f"Samples: {total_samples}")
    print(f"Elapsed sec: {elapsed:.6f}")
    print(f"Samples/sec: {samples_per_sec:.6f}")
    print(f"Last loss: {last_loss:.8f}")
    print(f"Last recon loss: {last_recon_loss:.8f}")
    print(f"Last cls loss: {last_cls_loss:.8f}")
    print(f"Accuracy: {metrics['accuracy']:.8f}")
    print(f"Precision macro: {metrics['precision_macro']:.8f}")
    print(f"Recall macro: {metrics['recall_macro']:.8f}")
    print(f"F1 macro: {metrics['f1_macro']:.8f}")

    if cfg.report_json is not None:
        report_payload: dict[str, Any] = {
            "device": cfg.device,
            "profile": cfg.profile,
            "dry_run": cfg.dry_run,
            "batch_size": cfg.batch_size,
            "steps": total_steps,
            "samples": total_samples,
            "elapsed_sec": elapsed,
            "samples_per_sec": samples_per_sec,
            "last_loss": last_loss,
            "last_recon_loss": last_recon_loss,
            "last_cls_loss": last_cls_loss,
            "class_names": class_names,
            "classification": metrics,
        }
        try:
            cfg.report_json.parent.mkdir(parents=True, exist_ok=True)
            cfg.report_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: failed writing report json: {exc}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
