# bc-trust-framework

Code for a trustworthiness-oriented evaluation pipeline for tabular clinical
classification: explanation-stability analysis, conformal prediction with a
reject option, decision-curve analysis, and multi-cohort evaluation.

I put this together to make the experiments in our breast-cancer study fully
reproducible. It is research code — readable rather than packaged — and each
script runs on its own. The small, reusable building blocks live in
`src/trust_metrics.py` and are unit-tested under `tests/`.

> **Status:** accompanies a manuscript under review. Code only for now; figures,
> numbers, and the paper reference will be added once it is published.

## Setup

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.12 (pinned versions in `requirements.txt`).

## Core module

`src/trust_metrics.py` contains the documented, importable functions used across
the experiments:
- `rank_stability`, `selection_stability` — the two explanation-stability axes
  (Kendall rank agreement; Kuncheva top-k consistency).
- `chance_correct`, `esi_star` — the chance-corrected stability index (ESI*),
  with its zero-null property.
- `conformal_thresholds`, `prediction_set`, `is_deferred`, `evaluate_conformal`
  — class-conditional conformal prediction with the reject option.

## Tests

```
pytest -q
```
Fast smoke tests (synthetic data, no external datasets) check that the stability
axes are bounded and correct at the extremes, that ESI* vanishes at the null,
and that conformal coverage meets its target while the reject option can abstain.
They also run automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

## Data

The four datasets are public (UCI) and not redistributed here. See
`data/README.md` for the IDs and the file names to drop into `data/`.

## Scripts (in `src/`)

Each one prints its results to the console; figures go to a local `eq/` folder.

| Script | What it does |
|---|---|
| `model_comparison.py` | cross-validated model comparison + significance tests |
| `multicohort_evaluation.py` | four-cohort evaluation: accuracy/AUC, stability, conformal coverage |
| `stability_index.py` | ESI components, bootstrap CIs |
| `stability_significance.py` | permutation null and Nogueira estimator for stability |
| `esistar_chance_corrected.py` | chance-corrected explanation-stability index (ESI*) |
| `esistar_permutation_test.py` | permutation significance test for ESI* (B = 299) |
| `subpopulation_transfer.py` | subpopulation model-transfer probe (no retraining) |
| `per_class_coverage.py` | per-class (Mondrian) conformal coverage |
| `competitor_protocol.py` | competitor pipelines under one common protocol |
| `attribution_robustness.py` | stability under SHAP / gain / permutation importance |
| `distribution_shift.py` | behaviour (coverage, deferral) under covariate shift |
| `decision_curve_analysis.py` | decision-curve / net-benefit analysis |
| `calibration_test.py` | calibration of stability-matched feature subsets |
| `figures_variance_stability.py` | protocol-variance and per-feature stability figures |
| `render_equations.py` | renders the equation images |
| `compute_shap_matrix.py` | generates `data/Phi_wdbc.npy` (run once before the transfer/shift scripts) |

A note: runs are seeded, so numbers are stable to ~3 decimals; tiny variation
can still come from multi-threaded boosting.

## Docker (optional)

```
docker build -t bc-trust .
docker run --rm bc-trust            # runs the smoke tests
```

## License

MIT — see `LICENSE`.

## Contact / citation

Hussein Ali Hussein Al Naffakh — University of Alkafeel — hussein.alnaffakh@alkafeel.edu.iq

A formal citation will be added when the paper appears.
