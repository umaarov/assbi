"""A trained neural intent classifier for the chatbot's NLU.

Architecture: bag-of-words features -> a small feed-forward neural network
(Linear -> ReLU -> Dropout -> Linear) -> softmax over intents. Trained with
cross-entropy on the synthetic dataset in :mod:`intent_data`, evaluated on a
held-out split (accuracy, macro-F1, confusion matrix).

This is the genuinely *trained* part of the assistant: the model learns to map a
natural-language question to an intent. The answer itself is then produced from
the analytics warehouse (grounded), so numbers stay correct.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_DEFAULT_DIR = Path("models/intent")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _build_net(vocab_size: int, n_classes: int, hidden: int = 64):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(vocab_size, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_classes),
    )


def _vectorize(texts: list[str], vocab: dict[str, int]):
    import torch

    X = torch.zeros(len(texts), len(vocab))
    for i, text in enumerate(texts):
        for tok in tokenize(text):
            j = vocab.get(tok)
            if j is not None:
                X[i, j] = 1.0
    return X


@dataclass
class TrainResult:
    model_path: Path
    dataset_path: Path
    metrics_path: Path
    confusion_path: Path | None
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    n_intents: int


def train(
    out_dir: str | Path = _DEFAULT_DIR,
    epochs: int = 200,
    hidden: int = 64,
    lr: float = 0.01,
    test_split: float = 0.2,
    seed: int = 0,
) -> TrainResult:
    """Train the intent classifier and save model + metrics + dataset."""
    import torch

    from . import intent_data

    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = intent_data.generate(seed=seed)
    dataset_path = intent_data.save_csv(rows, out / "dataset.csv")
    labels = sorted({intent for _, intent in rows})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # Vocab from the whole set (tiny, closed-domain).
    vocab: dict[str, int] = {}
    for text, _ in rows:
        for tok in tokenize(text):
            vocab.setdefault(tok, len(vocab))

    # Stratified-ish split: shuffle per class, hold out test_split of each.
    rng = __import__("random").Random(seed)
    by_class: dict[str, list[str]] = {}
    for text, intent in rows:
        by_class.setdefault(intent, []).append(text)
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    for intent, texts in by_class.items():
        rng.shuffle(texts)
        k = max(1, int(len(texts) * test_split))
        test_rows += [(t, intent) for t in texts[:k]]
        train_rows += [(t, intent) for t in texts[k:]]

    Xtr = _vectorize([t for t, _ in train_rows], vocab)
    ytr = torch.tensor([label_to_idx[i] for _, i in train_rows])
    Xte = _vectorize([t for t, _ in test_rows], vocab)
    yte = torch.tensor([label_to_idx[i] for _, i in test_rows])

    net = _build_net(len(vocab), len(labels), hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>3}/{epochs}  loss={loss.item():.4f}")

    # -- evaluate on held-out test set ------------------------------------
    net.eval()
    with torch.no_grad():
        pred = net(Xte).argmax(1)
    correct = (pred == yte).sum().item()
    accuracy = correct / len(yte)

    n = len(labels)
    confusion = [[0] * n for _ in range(n)]
    for p, t in zip(pred.tolist(), yte.tolist()):
        confusion[t][p] += 1
    # Per-class precision/recall/F1 -> macro-F1.
    f1s = []
    for c in range(n):
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in range(n)) - tp
        fn = sum(confusion[c]) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = sum(f1s) / n

    # -- save model artifact ---------------------------------------------
    model_path = out / "intent_model.pt"
    torch.save(
        {"state_dict": net.state_dict(), "vocab": vocab,
         "labels": labels, "hidden": hidden},
        model_path,
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "intents": labels,
        "per_intent_f1": {labels[c]: round(f1s[c], 4) for c in range(n)},
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    confusion_path = _plot_confusion(confusion, labels, out / "confusion_matrix.png")

    return TrainResult(
        model_path=model_path, dataset_path=dataset_path, metrics_path=metrics_path,
        confusion_path=confusion_path, accuracy=accuracy, macro_f1=macro_f1,
        n_train=len(train_rows), n_test=len(test_rows), n_intents=n,
    )


def _plot_confusion(confusion, labels, path: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Intent classifier — confusion matrix (test set)")
    for i in range(n):
        for j in range(n):
            if confusion[i][j]:
                ax.text(j, i, confusion[i][j], ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


class IntentClassifier:
    """Loads a trained model and predicts ``(intent, confidence)`` for a query."""

    def __init__(self, model_path: str | Path = _DEFAULT_DIR / "intent_model.pt") -> None:
        import torch

        self._torch = torch
        blob = torch.load(model_path, map_location="cpu", weights_only=False)
        self.vocab: dict[str, int] = blob["vocab"]
        self.labels: list[str] = blob["labels"]
        self.net = _build_net(len(self.vocab), len(self.labels), blob.get("hidden", 64))
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()

    def predict(self, text: str) -> tuple[str, float]:
        torch = self._torch
        x = _vectorize([text], self.vocab)
        with torch.no_grad():
            probs = torch.softmax(self.net(x)[0], dim=0)
        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])

    @classmethod
    def load_if_available(cls, model_path: str | Path = _DEFAULT_DIR / "intent_model.pt"):
        return cls(model_path) if Path(model_path).exists() else None
