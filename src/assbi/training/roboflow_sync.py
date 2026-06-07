"""Roboflow integration for the ASSBI custom dataset.

This implements the *hybrid* annotation workflow the assignment asks for:

    1. ``build-dataset`` samples frames from the Temple Bar footage and writes a
       local YOLO dataset with **auto-labels** (pseudo-labels from a teacher
       model).  See :mod:`assbi.training.dataset`.
    2. :func:`upload_dataset` pushes those frames **with their boxes already
       attached** to a Roboflow project, so in the Roboflow web app you only have
       to *review / correct* the annotations instead of drawing every box from
       scratch — then "Generate Version" (with augmentation) and the dataset is
       hosted, versioned and exportable.
    3. :func:`download_dataset` pulls a generated version back down in YOLOv8
       format via the Roboflow API, ready for ``cli train``.

Roboflow account creation, the annotation UI and "Generate Version" are manual
web steps (a browser, not this code) — see ``ROBOFLOW_GUIDE.md`` for the exact
clicks.  Everything that touches the codebase is here and fully automated.

The ``roboflow`` package is an optional, lazily-imported dependency (it is in
``requirements-full.txt``); none of the rest of the platform needs it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Single-class detector: every label index 0 is a person. Roboflow needs this
# map to interpret the bare class ids in our YOLO ``.txt`` files on upload.
_LABELMAP = {0: "person"}
_API_KEY_ENV = "ROBOFLOW_API_KEY"


@dataclass
class UploadStats:
    project: str
    uploaded: int
    with_annotations: int
    failed: int


@dataclass
class DownloadStats:
    location: Path
    data_yaml: Path
    version: int


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get(_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"No Roboflow API key. Pass --api-key, or set ${_API_KEY_ENV} "
            "(add it to your .env). Find it in Roboflow → your workspace → "
            "Settings → API Keys (the Private API Key)."
        )
    return key


def _project(api_key: str, workspace: str | None, project: str):
    try:
        from roboflow import Roboflow
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The 'roboflow' package is required for this command. Install it:\n"
            "  pip install roboflow\n"
            "(or pip install -r requirements-full.txt)"
        ) from exc

    rf = Roboflow(api_key=api_key)
    # workspace() with no arg uses the API key's default workspace.
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    return ws.project(project)


def upload_dataset(
    project: str,
    workspace: str | None = None,
    api_key: str | None = None,
    dataset_dir: str = "data/dataset",
    with_labels: bool = True,
    batch_name: str = "assbi-temple-bar",
    limit: int | None = None,
) -> UploadStats:
    """Upload the local YOLO dataset to a Roboflow project as pre-annotations.

    Args:
        project: Roboflow project id (the slug in the project URL).
        workspace: workspace id (omit to use the API key's default workspace).
        api_key: Roboflow *private* API key (or set ``$ROBOFLOW_API_KEY``).
        dataset_dir: local dataset root built by ``build-dataset``.
        with_labels: also upload the auto-labels so you only *review* them in
            Roboflow (the hybrid workflow). Set False to upload raw images only
            (fully manual annotation).
        batch_name: groups the upload in Roboflow's "Annotate" queue.
        limit: optionally cap how many images to upload (handy for a quick test).
    """
    key = _resolve_api_key(api_key)
    proj = _project(key, workspace, project)

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Build the dataset first:\n"
            "  python -m assbi.cli build-dataset --source data/source_video.mp4"
        )

    # Collect (image, label, split) for every frame across train/ and val/.
    items: list[tuple[Path, Path | None, str]] = []
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            ann = lbl_dir / f"{img.stem}.txt"
            has_ann = with_labels and ann.exists() and ann.stat().st_size > 0
            items.append((img, ann if has_ann else None, split))

    if not items:
        raise RuntimeError(f"No images found under {root}/images/. Nothing to upload.")
    if limit:
        items = items[:limit]

    total = len(items)
    print(f"Uploading {total} images to Roboflow project '{project}' "
          f"(batch '{batch_name}', labels={'on' if with_labels else 'off'}) …")

    uploaded = with_ann = failed = 0
    for i, (img, ann, split) in enumerate(items, 1):
        kwargs: dict = {
            "image_path": str(img),
            "split": split,
            "batch_name": batch_name,
            "num_retry_uploads": 3,
        }
        if ann is not None:
            kwargs["annotation_path"] = str(ann)
            # Tell Roboflow what class id 0 means (YOLO txt has no names).
            kwargs["annotation_labelmap"] = _LABELMAP
        try:
            proj.upload(**kwargs)
            uploaded += 1
            if ann is not None:
                with_ann += 1
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"  ! failed on {img.name}: {exc}")
        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} uploaded ({with_ann} with annotations, {failed} failed)…")

    print(f"\n✓ Uploaded {uploaded}/{total} images "
          f"({with_ann} carried auto-labels, {failed} failed).")
    print("Next: open Roboflow → Annotate → review/correct the boxes → "
          "Generate Version (add augmentation) → then `cli roboflow-download`.")
    return UploadStats(
        project=project, uploaded=uploaded, with_annotations=with_ann, failed=failed,
    )


def generate_version(
    project: str,
    workspace: str | None = None,
    api_key: str | None = None,
    augment: bool = True,
    resize: int = 416,
) -> int:
    """Generate a Roboflow dataset version (preprocessing + augmentation) via API.

    This is the programmatic equivalent of the web app's "Generate Version"
    button — it lets the whole pipeline run without the browser. Annotations
    already uploaded (even if shown as "unannotated"/unreviewed in the UI) are
    baked into the version, so the downloaded export has real labels.

    Returns the new version number (pass it to :func:`download_dataset`).
    """
    key = _resolve_api_key(api_key)
    proj = _project(key, workspace, project)

    preprocessing: dict = {"auto-orient": True}
    if resize:
        preprocessing["resize"] = {"width": resize, "height": resize, "format": "Stretch to"}
    augmentation: dict = {}
    if augment:
        augmentation = {
            "flip": {"horizontal": True, "vertical": False},
            "brightness": {"brighten": True, "darken": True, "percent": 18},
            "image": {"versions": 3},  # 3 augmented copies per train image
        }

    print(f"Generating a version of '{project}' "
          f"(resize={resize}, augment={'on' if augment else 'off'}) …")
    version = proj.generate_version(settings={
        "preprocessing": preprocessing,
        "augmentation": augmentation,
    })
    print(f"\n✓ Version {version} generation started on Roboflow.")
    print(f"Download it (give it ~1 min to build):\n"
          f"  python -m assbi.cli roboflow-download --project {project} --version {version}")
    return int(version)


def _fix_data_yaml(data_yaml: Path) -> None:
    """Rewrite a Roboflow ``data.yaml`` so Ultralytics resolves the splits.

    Roboflow exports use ``train: ../train/images`` (relative to a parent dir),
    which Ultralytics resolves against its own ``datasets`` dir, not the yaml —
    so training can't find the images. We pin an absolute ``path:`` at the
    dataset root and point ``train``/``val`` at the real folders that exist.
    """
    root = data_yaml.parent.resolve()

    def _first_existing(*candidates: str) -> str | None:
        for c in candidates:
            if (root / c).is_dir() and any((root / c).glob("*.jpg")):
                return c
        return None

    train = _first_existing("train/images", "images/train")
    val = _first_existing("valid/images", "val/images", "images/val") or train
    if not train:
        return  # unusual layout; leave the file as-is

    lines = ["names:", "- person", "nc: 1",
             f"path: {root.as_posix()}", f"train: {train}", f"val: {val}"]
    test = _first_existing("test/images", "images/test")
    if test:
        lines.append(f"test: {test}")
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_dataset(
    project: str,
    version: int,
    workspace: str | None = None,
    api_key: str | None = None,
    out_dir: str = "data/roboflow",
    fmt: str = "yolov8",
) -> DownloadStats:
    """Download a generated Roboflow version as a YOLOv8 dataset for training.

    Args:
        project: Roboflow project id.
        version: the version number you generated in Roboflow (1, 2, …).
        workspace: workspace id (omit for the key's default workspace).
        api_key: Roboflow private API key (or ``$ROBOFLOW_API_KEY``).
        out_dir: where to write the downloaded dataset.
        fmt: export format (``yolov8`` is what ``cli train`` expects).
    """
    key = _resolve_api_key(api_key)
    proj = _project(key, workspace, project)

    print(f"Downloading '{project}' v{version} as {fmt} → {out_dir}/ …")
    ver = proj.version(int(version))
    dataset = ver.download(fmt, location=out_dir, overwrite=True)

    location = Path(getattr(dataset, "location", out_dir))
    data_yaml = location / "data.yaml"
    if not data_yaml.exists():
        # Some exports nest the data.yaml; find it.
        found = list(location.rglob("data.yaml"))
        if found:
            data_yaml = found[0]
        else:
            raise RuntimeError(f"Download finished but no data.yaml under {location}.")

    _fix_data_yaml(data_yaml)
    print(f"\n✓ Downloaded to {location}/")
    print(f"Now train on it:\n  python -m assbi.cli train --data {data_yaml.as_posix()}")
    return DownloadStats(location=location, data_yaml=data_yaml, version=int(version))
