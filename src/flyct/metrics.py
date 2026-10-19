"""Evaluation metrics, overall and broken down by region (super class) and by cell-type size.

Every model returns its top-k predicted labels per neuron (best first) and, when it has one, the
probability of its top prediction. Accuracy and weighted F1 are what NTAC reports; macro F1 and
accuracy by type size show how rare classes fare; ECE checks that confidence means what it says.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SIZE_BUCKETS = [(2, 2, "2"), (3, 10, "3-10"), (11, 100, "11-100"), (101, np.inf, "gt100")]


def ece(correct: np.ndarray, conf: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: mean gap between confidence and accuracy over equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(conf))


def evaluate(y_true, topk, conf=None, region=None, class_size=None) -> dict:
    """Metrics for one set of scored neurons.

    y_true      (n,) true labels
    topk        (n, k) predicted labels, best first
    conf        (n,) probability of the top prediction, or None
    region      (n,) super class of each neuron, for per-region accuracy
    class_size  (n,) size of the neuron's true class among all labelled neurons, for per-size accuracy
    """
    y_true = np.asarray(y_true, dtype=object)
    topk = np.asarray(topk, dtype=object)
    pred = topk[:, 0]
    correct = pred == y_true
    m = {
        "n": len(y_true),
        "acc": correct.mean(),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
    }
    if topk.shape[1] >= 5:
        m["top5_acc"] = (topk[:, :5] == y_true[:, None]).any(axis=1).mean()
    if conf is not None:
        m["ece"] = ece(correct, np.asarray(conf, dtype=float))
    if region is not None:
        for r, acc in pd.Series(correct).groupby(np.asarray(region, dtype=object)).mean().items():
            m[f"acc_region_{r}"] = acc
    if class_size is not None:
        for lo, hi, name in SIZE_BUCKETS:
            sel = (class_size >= lo) & (class_size <= hi)
            if sel.any():
                m[f"acc_size_{name}"] = correct[sel].mean()
    return {k: float(v) for k, v in m.items()}
