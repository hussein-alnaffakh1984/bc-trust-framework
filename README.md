# bc-trust-framework

Code for a trustworthiness-oriented evaluation pipeline for tabular clinical
classification: explanation-stability analysis, conformal prediction with a
reject option, decision-curve analysis, and multi-cohort evaluation.

I put this together to make the experiments in our breast-cancer study fully
reproducible. It is research code — readable rather than packaged — and each
script runs on its own.

> **Status:** accompanies a manuscript under review. Code only for now; figures,
> numbers, and the paper reference will be added once it is published.

## Setup

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.12 (pinned versions in `requirements.txt`).

## Data

The four datasets are public (UCI) and not redistributed here. See
`data/README.md` for the IDs and the file names to drop into `data/`.

## Scripts (in `src/`)

Each one prints its results to the console; figures go to a local `eq/` folder.

| Script | What it does |
|---|---|
| `ensemble.py` | cross-validated model comparison + significance tests |
| `improve.py` | multi-cohort evaluation: accuracy/AUC, stability, conformal coverage |
| `fixes1.py`, `fixes2.py` | stability components, bootstrap CIs, permutation null |
| `esistar.py` | chance-corrected explanation-stability index |
| `perm999.py` | permutation significance test (B = 299) |
| `transfer.py` | subpopulation model-transfer probe (no retraining) |
| `close6.py` | per-class conformal coverage |
| `close7.py` | competitor pipelines under one protocol |
| `close10.py` | attribution-method comparison (SHAP / gain / permutation) |
| `close_bc.py` | behaviour under covariate shift |
| `close_dca.py` | decision-curve / net-benefit analysis |
| `predictive.py` | calibration of stability-matched feature subsets |
| `fig_stats.py` | protocol-variance and per-feature stability figures |
| `render_eqs.py` | renders the equation images |

A note: runs are seeded, so numbers are stable to ~3 decimals; tiny variation
can still come from multi-threaded boosting.

## License

MIT — see `LICENSE`.

## Contact / citation

Hussein Ali Hussein Al Naffakh — University of Alkafeel — hussein.alnaffakh@alkafeel.edu.iq

A formal citation will be added when the paper appears.
