"""
trust_metrics.py
Core, reusable functions for the trustworthiness framework: explanation-stability
indices (rank and selection consistency, chance-corrected ESI*) and class-conditional
conformal prediction with a reject option.

These are the small, well-defined building blocks used by the analysis scripts in
this folder. They are kept here, separately and documented, so they can be unit-tested
(see ../tests) independently of the full experiments.

Author: Hussein Ali Hussein Al Naffakh — University of Alkafeel.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau


# --------------------------------------------------------------------------- #
# Explanation-stability axes
# --------------------------------------------------------------------------- #
def rank_stability(Phi: np.ndarray) -> float:
    """Mean pairwise Kendall's tau between repeated importance vectors.

    Parameters
    ----------
    Phi : array, shape (R, p)
        R repetitions of a p-dimensional feature-importance vector.

    Returns
    -------
    float in [-1, 1]; 1.0 means identical rankings across repetitions,
    ~0.0 means rankings are unrelated (the chance level).
    """
    Phi = np.asarray(Phi, float)
    R = Phi.shape[0]
    if R < 2:
        raise ValueError("rank_stability needs at least 2 repetitions")
    taus = [kendalltau(Phi[a], Phi[b]).correlation
            for a in range(R) for b in range(a + 1, R)]
    return float(np.nanmean(taus))


def selection_stability(Phi: np.ndarray, k: int) -> float:
    """Kuncheva consistency index over the top-k selected features.

    Chance-corrected by construction: the expected value for independently
    drawn equal-size subsets is 0, a perfectly reproducible selection is 1.

    Parameters
    ----------
    Phi : array, shape (R, p)
    k   : size of the top-k set (0 < k < p)
    """
    Phi = np.asarray(Phi, float)
    R, p = Phi.shape
    if not 0 < k < p:
        raise ValueError("require 0 < k < p for the selection axis")
    tops = [set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]
    denom = k - k * k / p
    if denom == 0:                      # degenerate (k == p or tiny p)
        return float("nan")
    vals = [(len(tops[a] & tops[b]) - k * k / p) / denom
            for a in range(R) for b in range(a + 1, R)]
    return float(np.mean(vals))


def chance_correct(value: float, null_mean: float) -> float:
    """Centre a stability value on its permutation-null mean and rescale to [0, 1]."""
    if 1.0 - null_mean <= 0:
        return float("nan")
    return float(np.clip((value - null_mean) / (1.0 - null_mean), 0.0, 1.0))


def esi_star(rank_val: float, sel_val: float,
             null_rank_mean: float, null_sel_mean: float) -> float:
    """Chance-corrected Explanation Stability Index (ESI*).

    Geometric mean of the chance-corrected rank and selection axes. Lies in
    [0, 1]; equals 0 if either axis is at its chance level (zero-null property),
    and 1 only under perfect rank and selection reproducibility.
    """
    c_rank = chance_correct(rank_val, null_rank_mean)
    c_sel = chance_correct(sel_val, null_sel_mean)
    if np.isnan(c_rank) or np.isnan(c_sel):
        return float("nan")
    return float(np.sqrt(max(c_rank, 0.0) * max(c_sel, 0.0)))


# --------------------------------------------------------------------------- #
# Class-conditional (Mondrian) conformal prediction with a reject option
# --------------------------------------------------------------------------- #
def conformal_thresholds(p_cal: np.ndarray, y_cal: np.ndarray,
                         alpha: float = 0.10) -> dict:
    """Per-class nonconformity quantiles for split-conformal prediction.

    Parameters
    ----------
    p_cal : array, shape (n, 2) — calibrated class probabilities on a held-out set.
    y_cal : array, shape (n,)   — true labels (0/1) for the calibration set.
    alpha : target miscoverage (0.10 => 90% coverage).

    Returns
    -------
    dict {class: quantile}.
    """
    p_cal = np.asarray(p_cal, float)
    y_cal = np.asarray(y_cal, int)
    q = {}
    for c in (0, 1):
        scores = 1.0 - p_cal[y_cal == c, c]
        n = len(scores)
        if n == 0:
            q[c] = 1.0
            continue
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q[c] = float(np.quantile(scores, level))
    return q


def prediction_set(p_row: np.ndarray, q: dict) -> set:
    """Conformal prediction set for one example: classes whose nonconformity <= quantile."""
    return {c for c in (0, 1) if (1.0 - p_row[c]) <= q[c]}


def is_deferred(pred_set: set) -> bool:
    """Reject option: defer (abstain) when the set is not a single class."""
    return len(pred_set) != 1


def evaluate_conformal(p_test: np.ndarray, y_test: np.ndarray, q: dict) -> dict:
    """Empirical coverage, deferral (abstention) rate, and selective accuracy."""
    p_test = np.asarray(p_test, float)
    y_test = np.asarray(y_test, int)
    cov = defer = correct = singles = 0
    for i in range(len(y_test)):
        s = prediction_set(p_test[i], q)
        if y_test[i] in s:
            cov += 1
        if is_deferred(s):
            defer += 1
        else:
            singles += 1
            if next(iter(s)) == y_test[i]:
                correct += 1
    n = len(y_test)
    return {
        "coverage": cov / n,
        "deferral": defer / n,
        "selective_accuracy": correct / singles if singles else float("nan"),
    }
