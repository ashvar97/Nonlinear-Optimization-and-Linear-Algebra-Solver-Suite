#!/usr/bin/python3
"""Select k-NN's `k` and Parzen's bandwidth `h` by maximizing average held-out log-likelihood
over a range of candidate values.

Combines what used to be two separate, near-identical scripts
(`2_select_hyperparameters_knn.py` and `2_select_hyperparameters_parzen.py`).
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import make_ground_truth, build_cdf, draw_dataset, get_knn_pdf, get_parzen_pdf, \
    average_log_likelihood

show_plot = True

x_min, x_max = 0, 10
n_bins = 500
ground_truth_coefficients = (3, 0.3, 0.1)

num_samples = 200
sigma = 0.05

ground_truth_x, ground_truth_y = make_ground_truth(x_min, x_max, n_bins, ground_truth_coefficients)
cdf = build_cdf(ground_truth_y)

# --- k-NN: scan k ---
np.random.seed(41)
knn_training = draw_dataset(cdf, ground_truth_x, num_samples, sigma, x_min, x_max)
k_range = np.arange(3, 40)
knn_log_likelihoods = np.zeros(k_range.shape[0])
for idx, k in enumerate(k_range):
    knn_pdf, step_width = get_knn_pdf(k, knn_training, x_min, x_max, n_bins)
    ll = average_log_likelihood(cdf, ground_truth_x, sigma, x_min, x_max, knn_pdf, step_width)
    knn_log_likelihoods[idx] = ll
    print(f"k = \t{k}: \t {ll}")

best_k = k_range[np.argmax(knn_log_likelihoods)]
print(f"-> best k = {best_k}\n")

# --- Parzen: scan h ---
# h ranges from 0.1 to 3.9 in steps of 0.1; h_range holds it *10 so the sweep index stays integer.
np.random.seed(41)
parzen_training = draw_dataset(cdf, ground_truth_x, num_samples, sigma, x_min, x_max, clip=True)
h_range = np.arange(1, 40)
parzen_log_likelihoods = np.zeros(h_range.shape[0])
for idx, h10 in enumerate(h_range):
    parzen_pdf, step_width = get_parzen_pdf(h10 / 10.0, parzen_training, x_min, x_max, n_bins)
    ll = average_log_likelihood(cdf, ground_truth_x, sigma, x_min, x_max, parzen_pdf, step_width)
    parzen_log_likelihoods[idx] = ll
    print(f"h = \t{h10 / 10.0}: \t {ll}")

best_h = h_range[np.argmax(parzen_log_likelihoods)] / 10.0
print(f"-> best h = {best_h}")

if show_plot:
    fig, (ax_knn, ax_parzen) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_knn.plot(k_range, knn_log_likelihoods, "g", linewidth=3)
    ax_knn.axvline(best_k, color="r", linestyle="--", label=f"best k = {best_k}")
    ax_knn.set_xlim(k_range[0], k_range[-1])
    ax_knn.set_xlabel("k")
    ax_knn.set_ylabel("avg held-out log-likelihood")
    ax_knn.set_title("k-NN hyperparameter search")
    ax_knn.legend()

    ax_parzen.plot(h_range / 10, parzen_log_likelihoods, "g", linewidth=3)
    ax_parzen.axvline(best_h, color="r", linestyle="--", label=f"best h = {best_h}")
    ax_parzen.set_xlim(h_range[0] / 10, h_range[-1] / 10)
    ax_parzen.set_xlabel("h")
    ax_parzen.set_title("Parzen hyperparameter search")
    ax_parzen.legend()

    fig.tight_layout()
    plt.show()
