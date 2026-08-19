# Density Estimation & Ensembling

Python scripts exploring nonparametric density estimation (k-nearest-neighbours and Parzen/kernel
windows), likelihood-based hyperparameter selection, and two ensembling strategies (IID resampling
and bootstrap aggregating), on a synthetic 1D ground-truth distribution. Originally coursework for
a Pattern Analysis lecture, delivered as 8 scripts that were ~90% copy-pasted boilerplate (a KNN
and a Parzen variant of each of the 4 experiment stages below). Consolidated here into the shared
logic (`_common.py`) plus one script per stage that runs and compares both estimators, instead of
eight near-duplicate files.

The connection to the rest of this repo: hyperparameter selection here (choosing `k` or the
Parzen bandwidth `h` by maximizing held-out log-likelihood) is itself a small 1D optimization
problem, and the ensembling scripts follow the same "test the same idea against several methods"
structure as `optimization-suite/`.

Every script generates its own synthetic dataset from a fixed random seed and a hardcoded
ground-truth function (`_common.ground_truth_values`), so there is no external dataset or CLI to
configure — edit the constants near the top of a file (`num_samples`, `x_min`/`x_max`, `sigma`,
etc.) to change the experiment.

## Files

| File | What it does |
|---|---|
| `_common.py` | Shared, not runnable directly: ground-truth generation, dataset sampling, the k-NN and Parzen density estimators, and held-out log-likelihood evaluation. |
| `1_density_estimation.py` | Estimates the density of sampled data with both k-NN and Parzen estimators; plots both against the ground truth side by side. |
| `2_select_hyperparameters.py` | Scans `k` (k-NN) and `h` (Parzen) over a range, picks whichever maximizes average held-out log-likelihood. |
| `3_iid_ensembling.py` | For each candidate hyperparameter, averages several density estimates each fit on an *independent* draw from the distribution (IID ensembling), then scores the average. |
| `4_bagging_ensembling.py` | Same idea, but each ensemble member is fit on a *bootstrap resample of one shared dataset* (bagging) rather than an independent draw — contrast with `3_iid_ensembling.py`. |

## Requirements

```bash
pip install numpy matplotlib
```

## Usage

Each script runs standalone and opens a plot window at the end (`plt.show()`) — there is no
CLI; parameters live as constants near the top of each file:

```bash
python3 1_density_estimation.py
python3 2_select_hyperparameters.py
```

To run headless (e.g. over SSH, or in CI) without a display, set the matplotlib backend first:

```bash
MPLBACKEND=Agg python3 1_density_estimation.py
```

All four scripts have been run end-to-end and verified to complete without error (headless),
producing the same log-likelihood curves (within sampling noise) as the original 8-script
version before consolidation. One real compatibility bug was found and fixed along the way in
the original scripts: `np.Inf`, removed in NumPy 2.0, updated to `np.inf`.

## License

[MIT](../LICENSE)
