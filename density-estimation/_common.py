"""Shared ground-truth generation, sampling, density estimators, and evaluation used by the
four experiment scripts in this folder.

This used to be copy-pasted (with tiny variations) across all 8 original scripts -- every one
of `2_select_hyperparameters_*.py`, `3_iid_ensembling_*.py` and `4_bagging_ensembling_*.py`
carried its own near-identical copy of `draw_dataset`, `get_knn_pdf`/`get_parzen_pdf`, and
`evaluate_dataset`. Pulling it out here means the four remaining scripts contain only what's
actually specific to each experiment stage.
"""

import numpy as np


def ground_truth_values(x, coefficients):
    """Evaluate the synthetic ground-truth function at positions `x` (not yet normalized)."""
    a, b, c = coefficients
    return np.sin(x * a) * x * b + c * x + 2


def make_ground_truth(x_min, x_max, n_bins, coefficients):
    """Return (x, pdf) for the synthetic ground-truth density, normalized to sum to 1."""
    x = np.linspace(x_min, x_max, n_bins)
    y = ground_truth_values(x, coefficients)
    y = y / np.sum(y)
    return x, y


def build_cdf(ground_truth_y):
    """Cumulative sum of a (normalized) pdf, itself renormalized to end at exactly 1."""
    cdf = np.cumsum(ground_truth_y)
    return cdf / cdf[-1]


def draw_dataset(cdf, ground_truth_x, num_samples, sigma_noise, x_min, x_max, clip=False):
    """Sample `num_samples` points from the empirical cdf, plus homoscedastic Gaussian noise.

    `clip=True` clamps samples back into [x_min, x_max] -- required by the Parzen estimator
    below, which indexes a fixed-size bin array by sample position and would otherwise wrap or
    go out of range; the k-NN estimator only ever compares distances, so it doesn't need it.
    """
    u = np.sort(np.random.rand(num_samples))
    cdf_ptr = 0
    samples = np.zeros(num_samples)
    for u_pos in range(num_samples):
        while u[u_pos] > cdf[cdf_ptr]:
            cdf_ptr += 1
        samples[u_pos] = ground_truth_x[cdf_ptr]
    samples = samples + np.random.normal(0, sigma_noise, np.shape(samples))
    if clip:
        samples[samples < x_min] = x_min
        samples[samples > x_max] = x_max
    return np.sort(samples)


def get_knn_pdf(k, training, x_min, x_max, n_bins):
    """k-nearest-neighbours density estimate, evaluated on `n_bins` points over [x_min, x_max]."""
    step_width = (x_max - x_min) / n_bins
    knn_pdf = np.zeros(n_bins)
    n_samples = np.size(training)
    last_neighbor = 0
    first_neighbor = int(k - 1)

    def window(pos, lo, hi):
        return max(abs(pos - training[lo]), abs(pos - training[hi]))

    # all k nearest neighbors could initially be to the left of x_min; walk forward until the
    # window starting at the first sample position is as tight as it can get
    cur_v = window(training[0], last_neighbor, first_neighbor)
    next_v = window(training[0], last_neighbor + 1, first_neighbor + 1) if first_neighbor + 1 < n_samples else np.inf
    while cur_v > next_v:
        last_neighbor += 1
        first_neighbor += 1
        cur_v = next_v
        next_v = window(training[0], last_neighbor + 1, first_neighbor + 1) if first_neighbor + 1 < n_samples else np.inf

    for i in range(n_bins):
        cur_pos = training[0] + i * step_width
        cur_v = window(cur_pos, last_neighbor, first_neighbor)
        if first_neighbor + 1 < n_samples:
            next_v = window(cur_pos, last_neighbor + 1, first_neighbor + 1)
            # closer samples may have come into range as cur_pos advanced
            while cur_v > next_v:
                last_neighbor += 1
                first_neighbor += 1
                cur_v = next_v
                next_v = window(cur_pos, last_neighbor + 1, first_neighbor + 1) if first_neighbor + 1 < n_samples else np.inf
        knn_pdf[i] = k / ((cur_v + 0.001) * n_samples)

    return knn_pdf / np.sum(knn_pdf), step_width


def get_parzen_pdf(h, training, x_min, x_max, n_bins):
    """Parzen (rectangular kernel) window density estimate over [x_min, x_max]."""
    step_width = (x_max - x_min) / n_bins
    parzen_pdf = np.zeros(n_bins)
    sample_positions = ((training - x_min) / step_width).astype(np.uint)
    sample_positions[sample_positions >= n_bins] = n_bins - 1
    parzen_pdf[sample_positions] = 1
    parzen_window = np.ones(int(h / step_width))
    parzen_window = parzen_window / np.size(parzen_window)
    parzen_pdf = np.convolve(parzen_pdf, parzen_window, "same")
    parzen_pdf = parzen_pdf / np.sum(parzen_pdf)
    return parzen_pdf, step_width


def evaluate_dataset(testing, pdf, x_min, step_width):
    """Average log-likelihood of held-out samples `testing` under a discretized density `pdf`."""
    quality = 0.0
    for value in np.sort(testing):
        probe_pos = int((value - x_min) / step_width)
        if 0 <= probe_pos < pdf.shape[0]:
            likelihood = pdf[probe_pos]
            # unclean, but avoids -infinity for empty bins:
            quality += np.log(likelihood) if likelihood > 0 else np.log(0.001)
    return quality


def average_log_likelihood(cdf, ground_truth_x, sigma, x_min, x_max, pdf, step_width,
                            num_test_samples=50, num_runs=20):
    """Average held-out log-likelihood of `pdf` over `num_runs` independent test draws."""
    total = 0.0
    for _ in range(num_runs):
        testing = draw_dataset(cdf, ground_truth_x, num_test_samples, sigma, x_min, x_max)
        total += evaluate_dataset(testing, pdf, x_min, step_width)
    return total / num_runs
