#!/usr/bin/python3
"""Estimate the density of samples drawn from a synthetic 1D ground truth using both k-NN and
Parzen (kernel) window density estimation, and plot both estimates against the truth.

Combines what used to be two separate, near-identical scripts (`1_density_estimation_knn.py`
and `1_density_estimation_parzen.py`) into one, since the only real difference between them was
which estimator ran on the sampled data.
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import make_ground_truth, build_cdf, draw_dataset, get_knn_pdf, get_parzen_pdf

show_plot = True

x_min, x_max = 0, 10
n_bins = 500
ground_truth_coefficients = (3, 0.3, 0.1)

sigma = 0.05  # homoscedastic sampling noise

ground_truth_x, ground_truth_y = make_ground_truth(x_min, x_max, n_bins, ground_truth_coefficients)
cdf = build_cdf(ground_truth_y)

# k-NN: more samples, since its bin-free neighbor search tolerates sparser local density well
np.random.seed(45)
knn_samples = draw_dataset(cdf, ground_truth_x, num_samples=200, sigma_noise=sigma,
                            x_min=x_min, x_max=x_max)
knn_pdf, _ = get_knn_pdf(k=7, training=knn_samples, x_min=x_min, x_max=x_max, n_bins=n_bins)

# Parzen: fewer samples, clipped to bounds (required for its bin-indexed convolution)
np.random.seed(43)
parzen_samples = draw_dataset(cdf, ground_truth_x, num_samples=20, sigma_noise=sigma,
                               x_min=x_min, x_max=x_max, clip=True)
parzen_pdf, _ = get_parzen_pdf(h=0.1, training=parzen_samples, x_min=x_min, x_max=x_max, n_bins=n_bins)

if show_plot:
    fig, (ax_knn, ax_parzen) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    ax_knn.scatter(knn_samples, np.full_like(knn_samples, 0.005))
    ax_knn.plot(ground_truth_x, ground_truth_y, "g", linewidth=3, label="ground truth")
    ax_knn.plot(ground_truth_x, knn_pdf, "r", linewidth=3, label="k-NN estimate (k=7)")
    ax_knn.set_xlim(x_min, x_max)
    ax_knn.set_title("k-NN density estimate")
    ax_knn.legend()

    ax_parzen.scatter(parzen_samples, np.full_like(parzen_samples, 0.005))
    ax_parzen.plot(ground_truth_x, ground_truth_y, "g", linewidth=3, label="ground truth")
    ax_parzen.plot(ground_truth_x, parzen_pdf, "r", linewidth=3, label="Parzen estimate (h=0.1)")
    ax_parzen.set_xlim(x_min, x_max)
    ax_parzen.set_title("Parzen window density estimate")
    ax_parzen.legend()

    fig.tight_layout()
    plt.show()
