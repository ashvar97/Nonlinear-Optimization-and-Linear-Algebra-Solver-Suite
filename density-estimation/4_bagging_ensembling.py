#!/usr/bin/python3
"""Bagging ensembling: draw one base dataset, then average density estimates each fit on a
random subset (bootstrap resample) of that single dataset, rather than on independent draws
from the underlying distribution (contrast with `3_iid_ensembling.py`).

Combines what used to be two separate, near-identical scripts (`4_bagging_ensembling_knn.py`
and `4_bagging_ensembling_parzen.py`).
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import make_ground_truth, build_cdf, draw_dataset, get_knn_pdf, get_parzen_pdf, \
    average_log_likelihood

show_plot = True
num_ensemble_runs = 20
bagging_percentage = 0.5  # each bag draws num_samples out of num_samples/bagging_percentage

x_min, x_max = 0, 10
n_bins = 500
ground_truth_coefficients = (3, 0.3, 0.1)

num_samples = 200
sigma = 0.05

ground_truth_x, ground_truth_y = make_ground_truth(x_min, x_max, n_bins, ground_truth_coefficients)
cdf = build_cdf(ground_truth_y)


def bagging_ensemble_pdf(estimator, hyperparam, num_samples, clip):
    """Average `num_ensemble_runs` density estimates, each fit on a bootstrap resample of one
    shared base dataset (sized num_samples/bagging_percentage so each bag is a genuine subset)."""
    base_dataset = draw_dataset(cdf, ground_truth_x, int(num_samples / bagging_percentage), sigma,
                                 x_min, x_max, clip=clip)
    rng = np.random.default_rng()

    def bag():
        return rng.permutation(base_dataset)[0:num_samples]

    avg_pdf, step_width = estimator(hyperparam, bag())
    for _ in range(num_ensemble_runs - 1):
        pdf, step_width = estimator(hyperparam, bag())
        avg_pdf = avg_pdf + pdf
    return avg_pdf / num_ensemble_runs, step_width


# --- k-NN ---
np.random.seed(41)
k_range = np.arange(3, 40)
knn_log_likelihoods = np.zeros(k_range.shape[0])
for idx, k in enumerate(k_range):
    knn_pdf, step_width = bagging_ensemble_pdf(
        lambda h, training: get_knn_pdf(h, training, x_min, x_max, n_bins), k, num_samples, clip=False)
    ll = average_log_likelihood(cdf, ground_truth_x, sigma, x_min, x_max, knn_pdf, step_width)
    knn_log_likelihoods[idx] = ll
    print(f"k = \t{k}: \t {ll}")

# --- Parzen ---
np.random.seed(41)
h_range = np.arange(1, 40)
parzen_log_likelihoods = np.zeros(h_range.shape[0])
for idx, h10 in enumerate(h_range):
    parzen_pdf, step_width = bagging_ensemble_pdf(
        lambda h, training: get_parzen_pdf(h, training, x_min, x_max, n_bins), h10 / 10.0, num_samples, clip=True)
    ll = average_log_likelihood(cdf, ground_truth_x, sigma, x_min, x_max, parzen_pdf, step_width)
    parzen_log_likelihoods[idx] = ll
    print(f"h = \t{h10 / 10.0}: \t {ll}")

if show_plot:
    fig, (ax_knn, ax_parzen) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_knn.plot(k_range, knn_log_likelihoods, "g", linewidth=3)
    ax_knn.set_xlim(k_range[0], k_range[-1])
    ax_knn.set_xlabel("k")
    ax_knn.set_title("k-NN: bagging-ensembled")

    ax_parzen.plot(h_range / 10, parzen_log_likelihoods, "g", linewidth=3)
    ax_parzen.set_xlim(h_range[0] / 10, h_range[-1] / 10)
    ax_parzen.set_xlabel("h")
    ax_parzen.set_title("Parzen: bagging-ensembled")

    fig.tight_layout()
    plt.show()
