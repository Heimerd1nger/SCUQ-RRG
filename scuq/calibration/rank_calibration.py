"""Rank Calibration Error (RCE) — Section 2.2 / Eq. (1) of the paper.

References
----------
Wang et al., "Semantic Consistency-Based Uncertainty Quantification for
Factuality in Radiology Report Generation", NAACL 2025.
arXiv: 2412.04606
"""

from typing import List

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_rce(
    uncertainty_scores: List[float],
    factuality_scores: List[float],
    n_bins: int = 10,
) -> float:
    """Empirical Rank-Calibration Error (RCE).

    Args:
        uncertainty_scores: UQ scores for N reports.
        factuality_scores:  Corresponding factuality metric (e.g. GREEN score).
        n_bins:             Number of quantile bins B (default 10).

    Returns:
        rce: float, lower is better (0 = perfect calibration).
    """
    _, avg_uncertainty = _bin_stats(
        np.asarray(uncertainty_scores, dtype=float),
        np.asarray(factuality_scores, dtype=float),
        n_bins,
    )
    expected_rank = np.linspace(0, 1, n_bins)
    actual_rank = np.argsort(np.argsort(avg_uncertainty)) / max(n_bins - 1, 1)
    return float(np.mean(np.abs(actual_rank - expected_rank)))


def evaluate_calibration(
    uncertainty_scores: List[float],
    factuality_scores: List[float],
    n_bins: int = 10,
) -> dict:
    """Full calibration evaluation.

    Returns a dict with keys:
        ``rce``       — Empirical Rank-Calibration Error (lower is better)
        ``pearson_r`` — Pearson r (negative: uncertainty↑ → factuality↓)
        ``spearman_r``— Spearman r
        ``auroc``     — AUROC for detecting low-quality reports
    """
    u = np.asarray(uncertainty_scores, dtype=float)
    f = np.asarray(factuality_scores, dtype=float)

    rce = compute_rce(u, f, n_bins)
    pearson_r, _ = pearsonr(u, f)
    spearman_r, _ = spearmanr(u, f)
    auroc = _compute_auroc(u, f)

    return {
        "rce": rce,
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "auroc": float(auroc),
    }


def plot_calibration_curve(
    uncertainty_scores: List[float],
    factuality_scores: List[float],
    method_name: str = "VRO-GREEN",
    n_bins: int = 10,
):
    """Generate a calibration curve plot (bin mean uncertainty vs bin mean factuality).

    Returns the matplotlib Figure object so callers can save or display it.
    """
    import matplotlib.pyplot as plt

    u = np.asarray(uncertainty_scores, dtype=float)
    f = np.asarray(factuality_scores, dtype=float)
    expected_correctness, average_uncertainty = _bin_stats(u, f, n_bins)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.plot(average_uncertainty, expected_correctness, marker="o", label=method_name)
    ax.set_xlabel("Mean Uncertainty (per bin)", fontsize=12)
    ax.set_ylabel("Mean Factuality Score (per bin)", fontsize=12)
    ax.set_title("Rank Calibration Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bin_stats(uncertainty: np.ndarray, factuality: np.ndarray, n_bins: int):
    """Quantile-bin uncertainty values; return per-bin mean factuality and uncertainty."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(uncertainty, quantiles)
    bin_indices = np.digitize(uncertainty, bin_edges, right=True) - 1

    expected_correctness = np.zeros(n_bins)
    average_uncertainty = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(len(uncertainty)):
        b = int(np.clip(bin_indices[i], 0, n_bins - 1))
        expected_correctness[b] += factuality[i]
        average_uncertainty[b] += uncertainty[i]
        bin_counts[b] += 1

    expected_correctness /= np.maximum(bin_counts, 1)
    average_uncertainty /= np.maximum(bin_counts, 1)
    return expected_correctness, average_uncertainty


def _compute_auroc(uncertainty: np.ndarray, factuality: np.ndarray) -> float:
    """AUROC for detecting below-median factuality reports using uncertainty scores."""
    from sklearn.metrics import roc_auc_score
    threshold = np.median(factuality)
    labels = (factuality < threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, uncertainty))
