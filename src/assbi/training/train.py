"""Fine-tune YOLO on the dataset built by :mod:`assbi.training.dataset`.

Thin wrapper around Ultralytics' trainer that picks CPU-friendly defaults and
returns the path to the best weights plus the headline validation metrics — the
evidence a marker looks for (mAP, precision, recall, training curves under the
run directory).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainResult:
    best_weights: Path
    run_dir: Path
    metrics: dict


def train_model(
    data_yaml: str = "data/dataset/data.yaml",
    base: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 416,
    batch: int = 8,
    project: str = "runs/assbi",
    name: str = "finetune",
    device: str | None = None,
    patience: int = 20,
) -> TrainResult:
    """Fine-tune ``base`` on ``data_yaml`` and return the best weights + metrics.

    Defaults are tuned for CPU: a nano base, imgsz 416 and a modest batch keep a
    few hundred frames trainable in a sensible time. Raise ``epochs``/``imgsz``
    (or set ``device=0`` on a GPU machine) for higher accuracy.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required. pip install ultralytics") from exc

    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Build the dataset first: "
            "python -m assbi.cli build-dataset"
        )

    model = YOLO(base)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        # Absolute path so Ultralytics doesn't nest it under its own runs dir.
        project=str(Path(project).resolve()),
        name=name,
        device=device if device is not None else "cpu",
        patience=patience,
        plots=True,           # write training curves + confusion matrix (evidence)
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    # Pull the headline metrics if the validation results are available.
    metrics: dict = {}
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                  "metrics/precision(B)", "metrics/recall(B)"):
            if k in rd:
                metrics[k.split("/")[-1]] = round(float(rd[k]), 4)
    return TrainResult(best_weights=best, run_dir=run_dir, metrics=metrics)
