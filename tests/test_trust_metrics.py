"""
Smoke tests for the core trustworthiness metrics (trust_metrics.py).

These run on small synthetic data in a fraction of a second and need none of the
external datasets, so they are safe for continuous integration. They check the
properties the manuscript relies on: stability axes are bounded and behave
correctly at the extremes, ESI* has the zero-null property, and the conformal
layer achieves its target coverage while the reject option can abstain.

Run with:  pytest -q
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import trust_metrics as tm


# ----------------------------- stability axes ----------------------------- #
def test_rank_stability_perfect():
    Phi = np.tile(np.arange(10, dtype=float), (5, 1))   # identical rows
    assert tm.rank_stability(Phi) > 0.999


def test_rank_stability_bounds():
    rng = np.random.RandomState(0)
    Phi = rng.rand(8, 12)
    v = tm.rank_stability(Phi)
    assert -1.0 <= v <= 1.0


def test_selection_stability_perfect_and_bounds():
    Phi = np.tile(np.arange(12, dtype=float), (6, 1))
    assert abs(tm.selection_stability(Phi, k=4) - 1.0) < 1e-9
    rng = np.random.RandomState(1)
    v = tm.selection_stability(rng.rand(6, 12), k=4)
    assert -0.5 <= v <= 1.0


def test_chance_correct_range():
    assert tm.chance_correct(0.9, 0.5) == 0.8
    assert tm.chance_correct(0.5, 0.5) == 0.0       # at chance -> 0
    assert tm.chance_correct(0.4, 0.5) == 0.0       # below chance clipped to 0


# --------------------------------- ESI* ------------------------------------ #
def test_esi_star_unit_interval():
    v = tm.esi_star(0.94, 0.97, 0.87, 0.76)
    assert 0.0 <= v <= 1.0


def test_esi_star_zero_null_property():
    # if an axis sits at its null mean, ESI* must vanish
    assert tm.esi_star(0.87, 0.97, 0.87, 0.76) == 0.0


def test_esi_star_perfect():
    assert abs(tm.esi_star(1.0, 1.0, 0.5, 0.0) - 1.0) < 1e-9


# ----------------------------- conformal layer ----------------------------- #
def _toy_probs(n=2000, seed=0):
    """Well-separated synthetic two-class problem with calibrated-ish scores."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    p1 = np.clip(0.5 + (y - 0.5) * rng.uniform(0.3, 0.9, n) + rng.normal(0, 0.08, n), 0.01, 0.99)
    P = np.column_stack([1 - p1, p1])
    return P, y


def test_conformal_coverage_meets_target():
    P, y = _toy_probs(seed=2)
    half = len(y) // 2
    q = tm.conformal_thresholds(P[:half], y[:half], alpha=0.10)
    res = tm.evaluate_conformal(P[half:], y[half:], q)
    # empirical coverage should be at least roughly the 90% target (allow slack)
    assert res["coverage"] >= 0.85
    assert 0.0 <= res["deferral"] <= 1.0


def test_reject_option_can_abstain():
    # a maximally ambiguous point should not yield a confident singleton
    q = {0: 0.6, 1: 0.6}
    s = tm.prediction_set(np.array([0.5, 0.5]), q)
    assert tm.is_deferred(s)            # both classes admitted -> defer


def test_reject_option_confident_singleton():
    q = {0: 0.3, 1: 0.3}
    s = tm.prediction_set(np.array([0.02, 0.98]), q)
    assert s == {1} and not tm.is_deferred(s)
