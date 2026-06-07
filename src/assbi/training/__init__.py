"""Model-training tools: build a labelled image dataset from footage and
fine-tune YOLO on it.

This is the "make your own dataset and train the model yourself" deliverable.
The platform ships with a COCO-pretrained model, but these tools let you create
a site-specific dataset (frames sampled from the actual camera, auto-annotated
and reviewable) and fine-tune the detector on it for better local accuracy.
"""
from .dataset import DatasetStats, build_dataset
from .roboflow_sync import (
    DownloadStats,
    UploadStats,
    download_dataset,
    generate_version,
    upload_dataset,
)
from .train import TrainResult, train_model

__all__ = [
    "build_dataset", "DatasetStats",
    "train_model", "TrainResult",
    "upload_dataset", "UploadStats",
    "generate_version",
    "download_dataset", "DownloadStats",
]
