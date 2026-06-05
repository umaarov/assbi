"""Build a YOLO-format detection dataset from a video.

Pipeline:
    1. Sample N frames evenly across the clip (so the dataset spans the whole
       footage, not one moment).
    2. Auto-annotate each frame with a *stronger* pretrained model (the
       "teacher", e.g. YOLOv8s/m) — this produces pseudo-labels for the person
       class in YOLO ``txt`` format. Frames that contain people are preferred so
       the dataset is positive-rich.
    3. Split into train/val and write a ``data.yaml`` ready for ``yolo train``.

The pseudo-labels are a fast, reproducible starting point. For an honest
submission, open the dataset in a labelling tool (e.g. Roboflow / LabelImg) and
spot-check / correct a sample before training — the layout written here is the
standard one those tools import.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# COCO class index for "person" (the only class we keep).
_PERSON_COCO_ID = 0


@dataclass
class DatasetStats:
    root: Path
    data_yaml: Path
    train_images: int
    val_images: int
    total_labels: int
    frames_with_people: int


def build_dataset(
    source: str = "data/source_video.mp4",
    out_dir: str = "data/dataset",
    n_frames: int = 500,
    val_split: float = 0.2,
    label_model: str = "yolov8s.pt",
    conf: float = 0.30,
    imgsz: int = 640,
    seed: int = 0,
) -> DatasetStats:
    """Extract, auto-label and split ``n_frames`` frames from ``source``.

    Args:
        source: video file to sample (a recorded clip; not a live URL).
        out_dir: dataset root to create (images/ + labels/ + data.yaml).
        n_frames: number of labelled frames to keep.
        val_split: fraction held out for validation.
        label_model: pretrained "teacher" weights used to auto-annotate.
        conf: confidence threshold for keeping a detection as a label.
        imgsz: inference size for the teacher (640 gives better labels).
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python is required. pip install -r requirements-full.txt") from exc
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required. pip install ultralytics") from exc

    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}. Record/download footage first.")

    root = Path(out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError("Video reports zero frames; cannot sample it.")

    # Sample up to 3x candidates evenly, so we can prefer frames that actually
    # contain people while still ending up with ~n_frames kept.
    n_candidates = min(total, max(n_frames, n_frames * 3))
    step = max(1, total // n_candidates)
    candidate_indices = list(range(0, total, step))

    teacher = YOLO(label_model)
    val_every = max(2, round(1 / val_split)) if val_split > 0 else 0

    kept = 0
    kept_with_people = 0
    total_labels = 0
    val_images = 0
    print(f"Sampling up to {n_frames} frames from {total:,} (every ~{step} frames), "
          f"auto-labelling with {label_model} …")

    for idx in candidate_indices:
        if kept >= n_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, image = cap.read()
        if not ok or image is None:
            continue

        res = teacher.predict(image, conf=conf, imgsz=imgsz, classes=[_PERSON_COCO_ID], verbose=False)[0]
        h, w = image.shape[:2]
        lines: list[str] = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        has_people = bool(lines)
        # Once we've gathered enough people-frames, allow a few empties through to
        # the very end as background negatives; otherwise skip empty frames.
        if not has_people and kept_with_people < n_frames * 0.7:
            continue

        split = "val" if (val_every and kept % val_every == 0) else "train"
        name = f"frame_{idx:07d}"
        cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), image)
        (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

        kept += 1
        total_labels += len(lines)
        if has_people:
            kept_with_people += 1
        if split == "val":
            val_images += 1
        if kept % 50 == 0:
            print(f"  kept {kept}/{n_frames} frames ({total_labels} labels so far)…")

    cap.release()

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "# Auto-generated by `assbi build-dataset`. Single-class person detector.\n"
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )

    print(f"\n✓ Dataset ready at {root}/ — {kept} images "
          f"({kept - val_images} train / {val_images} val), {total_labels} person labels.")
    return DatasetStats(
        root=root, data_yaml=data_yaml,
        train_images=kept - val_images, val_images=val_images,
        total_labels=total_labels, frames_with_people=kept_with_people,
    )
