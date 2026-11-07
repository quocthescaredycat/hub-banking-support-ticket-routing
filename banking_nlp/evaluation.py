"""Multiclass metrics, bootstrap intervals, confusions and latency.

All metrics use the fixed list of 77 label IDs. A missing or invalid prediction
(any value outside 0..76, e.g. -1) counts as an error and stays in the
denominator; it is never dropped silently.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

N_LABELS = 77


def sanitize_predictions(y_pred, n_labels: int = N_LABELS) -> tuple[np.ndarray, int]:
    """Map missing/invalid predictions to -1 and return how many there were."""
    arr = np.array(pd.to_numeric(pd.Series(y_pred), errors="coerce").fillna(-1), dtype=float, copy=True)
    bad = ~np.isin(arr, np.arange(n_labels))
    arr[bad] = -1
    return arr.astype(int), int(bad.sum())


def topk_accuracy(y_true, scores: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return float(np.mean([t in row for t, row in zip(np.asarray(y_true), topk)]))


def classification_metrics(y_true, y_pred, scores: np.ndarray | None = None,
                           n_labels: int = N_LABELS) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred, n_invalid = sanitize_predictions(y_pred, n_labels)
    labels = np.arange(n_labels)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "invalid_predictions": n_invalid,
    }
    if scores is not None:
        out["top3_accuracy"] = topk_accuracy(y_true, scores, 3)
    return out


def macro_f1(y_true, y_pred, n_labels: int = N_LABELS) -> float:
    return float(precision_recall_fscore_support(y_true, y_pred, labels=np.arange(n_labels),
                                                 average="macro", zero_division=0)[2])


def bootstrap_indices(n: int, n_boot: int = 1000, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(y_true, y_pred, idx: np.ndarray, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for macro-F1 and accuracy over examples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = np.array([macro_f1(y_true[i], y_pred[i]) for i in idx])
    accs = np.array([np.mean(y_true[i] == y_pred[i]) for i in idx])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"macro_f1_ci": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
            "accuracy_ci": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))]}


def paired_bootstrap_delta(y_true, pred_a, pred_b, idx: np.ndarray, alpha: float = 0.05) -> dict:
    """Macro-F1(b) - macro-F1(a) on identical resamples (paired)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    point = macro_f1(y_true, pred_b) - macro_f1(y_true, pred_a)
    deltas = np.array([macro_f1(y_true[i], pred_b[i]) - macro_f1(y_true[i], pred_a[i]) for i in idx])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"delta_macro_f1": float(point),
            "delta_ci": [float(np.percentile(deltas, lo)), float(np.percentile(deltas, hi))],
            "p_delta_le_0": float(np.mean(deltas <= 0)),
            "n_disagreements": int(np.sum(pred_a != pred_b))}


def per_class_table(y_true, y_pred, label_names: Sequence[str]) -> pd.DataFrame:
    labels = np.arange(len(label_names))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"label_id": labels, "label_name": list(label_names), "support": s,
                         "precision": p, "recall": r, "f1": f}).sort_values("f1")


def top_confusions(y_true, y_pred, label_names: Sequence[str], k: int = 15) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None, kind="stable")[:k]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, cm.shape)
        if cm[i, j] == 0:
            break
        rows.append({"true": label_names[i], "predicted": label_names[j], "count": int(cm[i, j])})
    return pd.DataFrame(rows)


def latency_batch_one(predict: Callable[[list[str]], object], texts: Sequence[str],
                      warmup: int = 20, n: int = 200) -> dict:
    """Warm batch-one latency over a fixed list of inputs (first ``n`` texts)."""
    texts = list(texts)
    for t in texts[:warmup]:
        predict([t])
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        predict([t])
        times.append(time.perf_counter() - t0)
    ms = np.array(times) * 1000
    return {"n": len(ms), "warmup": warmup, "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(ms.mean())}
