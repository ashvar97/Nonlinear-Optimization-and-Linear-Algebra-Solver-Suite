# Density Estimation & Ensembling

Standalone Python scripts exploring nonparametric density estimation (k-nearest-neighbours and
Parzen/kernel windows), likelihood-based hyperparameter selection, and two ensembling strategies
(IID resampling and bootstrap aggregating), on a synthetic 1D ground-truth distribution. Originally
coursework for a Pattern Analysis lecture.

The connection to the rest of this repo: hyperparameter selection here (choosing `k` or the
Parzen bandwidth `h` by maximizing held-out log-likelihood) is itself a small 1D optimization
problem, and the ensembling scripts follow the same "test the same idea against several methods"
structure as `optimization-suite/`.

Every script is self-contained: it generates its own synthetic dataset from a fixed random seed
and a hardcoded ground-truth function (`get_function_values`), so there is no external dataset or
CLI to configure — edit the constants near the top of a file (`num_samples`, `x_min`/`x_max`,
`sigma_noise`, etc.) to change the experiment.

## Scripts, in progression order

| Script | What it does |
|---|---|
| `1_density_estimation_knn.py` | Estimates the density of samples drawn from the synthetic ground truth using k-NN density estimation; plots the estimate against the truth. |
| `1_density_estimation_parzen.py` | Same, using a Parzen (Gaussian kernel) window estimator. |
| `2_select_hyperparameters_knn.py` | Splits data into train/test, scans `k` over a range, picks the `k` that maximizes held-out log-likelihood. |
| `2_select_hyperparameters_parzen.py` | Same, scanning the Parzen bandwidth `h`. |
| `3_iid_ensembling_knn.py` | Trains several k-NN estimators on independent draws from the same distribution and averages them (an IID ensemble). |
| `3_iid_ensembling_parzen.py` | Same, with Parzen estimators. |
| `4_bagging_ensembling_knn.py` | Bootstrap-aggregates (bagging) k-NN estimators trained on resamples of a single dataset, rather than independent draws. |
| `4_bagging_ensembling_parzen.py` | Same, with Parzen estimators. |

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

Each script runs standalone and opens a plot window at the end (`plt.show()`) — there is no
CLI, no `--data`/`--k` flags; parameters live as constants near the top of each file:

```bash
python3 1_density_estimation_knn.py
python3 2_select_hyperparameters_parzen.py
```

To run headless (e.g. over SSH, or in CI) without a display, set the matplotlib backend first:

```bash
MPLBACKEND=Agg python3 1_density_estimation_knn.py
```

All eight scripts have been run end-to-end and verified to complete without error (headless).
One compatibility fix was needed and applied: the four `*_knn.py` scripts used `np.Inf`, which
was removed in NumPy 2.0 — updated to `np.inf`.

## License

[MIT](../LICENSE)
