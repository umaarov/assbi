"""Professional intent classifier via transfer learning.

Instead of bag-of-words, questions are embedded with a **pretrained sentence
transformer** (MiniLM) whose 384-d vectors already capture meaning. A small
neural head (Linear -> ReLU -> Dropout -> Linear) is then *trained* on top to map
those embeddings to intents. This is the standard production recipe (frozen
pretrained encoder + trained task head) and generalises far better to unseen
wording than keyword/BoW approaches.

Evaluation is rigorous: a stratified train/test split, 5-fold cross-validation
for a robust accuracy estimate, and a separate hand-written *hard* paraphrase
set to measure honest generalisation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DIR = Path("models/intent")
_HEAD_PATH = _DIR / "nlu_head.pt"

# Process-wide cache so the encoder is loaded once, not per query.
_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(_ENCODER_NAME)
    return _ENCODER


def _build_head(in_dim: int, n_classes: int, hidden: int = 128):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_classes),
    )


@dataclass
class NLUTrainResult:
    head_path: Path
    metrics_path: Path
    confusion_path: Path | None
    dataset_path: Path
    test_accuracy: float
    macro_f1: float
    cv_mean: float
    cv_std: float
    hard_accuracy: float
    n_train: int
    n_test: int
    n_intents: int


def train(out_dir: str | Path = _DIR, epochs: int = 300, hidden: int = 128,
          lr: float = 0.01, test_split: float = 0.2, seed: int = 0) -> NLUTrainResult:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.model_selection import cross_val_score, train_test_split

    from . import intent_data

    torch.manual_seed(seed)
    np.random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = intent_data.generate(seed=seed)
    dataset_path = intent_data.save_csv(rows, out / "dataset.csv")
    texts = [t for t, _ in rows]
    labels = sorted({i for _, i in rows})
    lab2idx = {lab: k for k, lab in enumerate(labels)}
    y = np.array([lab2idx[i] for _, i in rows])

    print(f"Embedding {len(texts)} questions with {_ENCODER_NAME} …")
    enc = _encoder()
    X = enc.encode(texts, batch_size=64, show_progress_bar=False,
                   normalize_embeddings=True)
    X = np.asarray(X, dtype="float32")

    # 5-fold CV on a linear probe for a robust accuracy estimate.
    print("5-fold cross-validation …")
    cv = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5)
    cv_mean, cv_std = float(cv.mean()), float(cv.std())

    # Stratified hold-out split.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y)

    # Train the neural head.
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    head = _build_head(X.shape[1], len(labels), hidden)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    head.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 75 == 0:
            print(f"  epoch {ep+1:>3}/{epochs}  loss={loss.item():.4f}")

    head.eval()
    with torch.no_grad():
        pred = head(torch.tensor(Xte)).argmax(1).numpy()
    test_acc = float(accuracy_score(yte, pred))
    macro_f1 = float(f1_score(yte, pred, average="macro"))

    # Honest generalisation: the hand-written hard paraphrase set.
    hard = intent_data.hard_eval()
    Xh = np.asarray(enc.encode([t for t, _ in hard], normalize_embeddings=True),
                    dtype="float32")
    yh = np.array([lab2idx[i] for _, i in hard])
    with torch.no_grad():
        ph = head(torch.tensor(Xh)).argmax(1).numpy()
    hard_acc = float(accuracy_score(yh, ph))

    # -- save artifacts ---------------------------------------------------
    torch.save({"state_dict": head.state_dict(), "labels": labels,
                "hidden": hidden, "in_dim": X.shape[1],
                "encoder": _ENCODER_NAME}, out / "nlu_head.pt")

    report = classification_report(yte, pred, target_names=labels,
                                   output_dict=True, zero_division=0)
    metrics = {
        "approach": "transfer learning (MiniLM embeddings + neural head)",
        "encoder": _ENCODER_NAME,
        "test_accuracy": round(test_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "cv5_mean": round(cv_mean, 4),
        "cv5_std": round(cv_std, 4),
        "hard_eval_accuracy": round(hard_acc, 4),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "intents": labels,
        "per_intent_f1": {lab: round(report[lab]["f1-score"], 4) for lab in labels},
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    confusion_path = _plot_confusion(confusion_matrix(yte, pred), labels,
                                     out / "confusion_matrix.png")

    return NLUTrainResult(
        head_path=out / "nlu_head.pt", metrics_path=metrics_path,
        confusion_path=confusion_path, dataset_path=dataset_path,
        test_accuracy=test_acc, macro_f1=macro_f1, cv_mean=cv_mean, cv_std=cv_std,
        hard_accuracy=hard_acc, n_train=len(ytr), n_test=len(yte), n_intents=len(labels),
    )


def _plot_confusion(cm, labels, path: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Intent classifier — confusion matrix (test set)")
    for i in range(n):
        for j in range(n):
            if cm[i][j]:
                ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


class NLUClassifier:
    """Loads the pretrained encoder + trained head; predicts ``(intent, conf)``."""

    def __init__(self, head_path: str | Path = _HEAD_PATH) -> None:
        import torch

        self._torch = torch
        blob = torch.load(head_path, map_location="cpu", weights_only=False)
        self.labels = blob["labels"]
        self.enc = _encoder()
        self.head = _build_head(blob["in_dim"], len(self.labels), blob.get("hidden", 128))
        self.head.load_state_dict(blob["state_dict"])
        self.head.eval()

    def predict(self, text: str) -> tuple[str, float]:
        torch = self._torch
        import numpy as np

        vec = np.asarray(self.enc.encode([text], normalize_embeddings=True), dtype="float32")
        with torch.no_grad():
            probs = torch.softmax(self.head(torch.tensor(vec))[0], dim=0)
        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])

    @classmethod
    def load_if_available(cls, head_path: str | Path = _HEAD_PATH):
        if not Path(head_path).exists():
            return None
        try:
            return cls(head_path)
        except Exception:
            return None
