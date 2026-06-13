# Trustworthy Tabular Classification — Code

Research code for a trustworthiness-oriented evaluation pipeline for tabular
clinical classification, combining explanation-stability analysis, distribution-free
uncertainty with a reject option, and multi-cohort evaluation.

> **Status.** This repository accompanies a manuscript currently under peer review.
> It contains **code only**. Numerical results, figures, and the paper reference will
> be added when the work is published. Until then please treat the methods here as
> work in progress.

---

## Components

The code implements and evaluates the following, independently runnable, pieces:

- **Explanation-stability index.** A permutation-calibrated index built from the
  rank and selection consistency of repeated SHAP attributions, with each axis
  centred on a label-permutation null.
- **Conformal uncertainty + reject option.** Class-conditional (Mondrian) conformal
  prediction with repeated cross-conformal calibration and an abstention rule for
  ambiguous cases.
- **Protocol-variance analysis.** Quantifies how much single train/test-split
  accuracy varies for a fixed model across many random seeds.
- **Fair competitor comparison.** Re-implements several feature-selection /
  classifier pipelines under one repeated cross-validation protocol.
- **Decision-curve analysis.** Net-benefit computation across threshold
  probabilities versus treat-all / treat-none baselines.
- **Behaviour under covariate shift.** Tracks coverage, deferral, and selective
  accuracy as controlled noise is added at test time.
- **Subpopulation model-transfer probe.** Trains on one covariate-defined
  subpopulation and evaluates on a disjoint one without retraining.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Python 3.12 with the pinned versions in `requirements.txt`.

## Data

Datasets are public and are **not** bundled here. See [`data/README.md`](data/README.md)
for the UCI sources and the expected file names; place them in `data/`.

## Usage

Each script in `src/` is self-contained and prints what it computes to stdout
(any figures are written to a local `eq/` folder, which is git-ignored). Example:

```bash
python src/ensemble.py        # cross-validated model comparison
python src/improve.py         # multi-cohort evaluation with conformal calibration
```

| Script | Computes |
|---|---|
| `ensemble.py` | cross-validated classifier comparison with significance tests |
| `improve.py` | multi-cohort evaluation: accuracy/AUC, stability, conformal coverage |
| `fixes1.py`, `fixes2.py` | stability-index components, bootstrap intervals, permutation null |
| `esistar.py` | chance-corrected explanation-stability index |
| `perm999.py` | permutation significance test for the stability index |
| `transfer.py` | subpopulation model-transfer probe (no retraining) |
| `close6.py` | per-class conformal coverage |
| `close7.py` | competitor pipelines under one protocol |
| `close10.py` | attribution-method comparison |
| `close_bc.py` | behaviour under controlled covariate shift |
| `close_dca.py` | decision-curve / net-benefit analysis |
| `predictive.py` | calibration comparison of feature subsets |
| `fig_stats.py` | protocol-variance and per-feature stability figures |
| `render_eqs.py` | renders equation images |

## Reproducibility

Randomness is seeded and library versions are pinned, so runs are deterministic up
to minor third-decimal variation from multi-threaded gradient boosting.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

A citation will be added upon publication. If you use this code before then,
please contact the author.
