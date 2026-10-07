"""Metrics on ONE shared test split: macro-F1, accuracy, ROC-AUC, bootstrap intervals, McNemar tests, slices."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, roc_auc_score


def macro_f1(y_true, y_pred) -> float:
    """Macro F1 over the two labels, from confusion counts (fast enough for the bootstrap)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = []
    for c in (0, 1):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = [macro_f1(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on the examples where exactly one of the two models is correct."""
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def scores(y_true, proba, n_boot: int = 1000, seed: int = 0) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    lo, hi = bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
    out = {
        "n": int(len(y_true)),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci95": [round(lo, 4), round(hi, 4)],
        "accuracy": float((y_true == y_pred).mean()),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
    return out


def slice_scores(y_true, y_pred, masks: dict[str, np.ndarray]) -> dict:
    """Macro-F1 and accuracy on each named subset of the test split (for error analysis)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        out[name] = {"n": int(mask.sum()), "macro_f1": macro_f1(y_true[mask], y_pred[mask]),
                     "accuracy": float((y_true[mask] == y_pred[mask]).mean())}
    return out
