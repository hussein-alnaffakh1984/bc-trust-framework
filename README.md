# Datasets

This project uses four publicly available datasets. They are **not redistributed
here**; download them directly from the UCI Machine Learning Repository and place
the files in this folder with the names below.

| File name (place here) | Source |
|---|---|
| `wdbc.csv` | Breast Cancer Wisconsin (Diagnostic) — UCI ID 17 |
| `wbc_selva.csv` | Breast Cancer Wisconsin (Original) — UCI ID 15 |
| `coimbra.csv` | Breast Cancer Coimbra — UCI ID 451 |
| `mammo.csv` | Mammographic Mass — UCI ID 161 |

`Phi_wdbc.npy` (a cached importance matrix used by a few scripts) is produced by
running `python src/compute_shap_matrix.py` once after `wdbc.csv` is in place.

Please cite the original UCI sources in any derived work.
