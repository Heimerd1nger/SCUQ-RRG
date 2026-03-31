"""Report- and sentence-level abstention utilities."""

from typing import List, Tuple
import random

import numpy as np
import pandas as pd


def get_abstention_indices(uncertainty_scores: np.ndarray, abstention_fraction: float) -> np.ndarray:
    """Return indices of the (1 - abstention_fraction) least uncertain reports.

    Args:
        uncertainty_scores:  Array of per-report uncertainty values.
        abstention_fraction: Fraction to withhold, e.g. 0.20 = 20 %.

    Returns:
        Indices of retained reports, sorted by ascending uncertainty.
    """
    n_keep = max(1, int(len(uncertainty_scores) * (1.0 - abstention_fraction)))
    return np.argsort(uncertainty_scores)[:n_keep]


def prune_sentences(
    reports: List[str],
    sentence_uncertainties: List[List[float]],
    threshold: float,
) -> List[str]:
    """Remove high-uncertainty sentences from each report.

    Sentences with uncertainty > threshold are pruned. At least one sentence
    per report is always retained (the least uncertain).

    Args:
        reports:               List of original reports.
        sentence_uncertainties: Per-sentence uncertainty lists (one per report).
        threshold:             Sentences above this value are removed.

    Returns:
        List of pruned report strings.
    """
    pruned = []
    for report, sent_unc in zip(reports, sentence_uncertainties):
        sentences = [s for s in report.split(". ") if s.strip()]
        if not sentences:
            pruned.append(report)
            continue

        kept = [s for s, u in zip(sentences, sent_unc) if u <= threshold]
        if not kept:
            # Always retain the least uncertain sentence
            min_idx = int(np.argmin(sent_unc))
            kept = [sentences[min_idx]]

        pruned.append(". ".join(kept))
    return pruned


def prune_sentences_random(
    reports: List[str],
    sentence_uncertainties: List[List[float]],
    seed: int = 42,
) -> List[str]:
    """Random-baseline sentence pruning: remove the same number of sentences as
    the VRO-RadGraph method would, but chosen at random.

    Args:
        reports:               List of original reports.
        sentence_uncertainties: Per-sentence uncertainty lists (used only to
                                determine how many sentences to prune per report).
        seed:                  Random seed for reproducibility.

    Returns:
        List of pruned report strings.
    """
    rng = random.Random(seed)
    pruned = []
    for report, sent_unc in zip(reports, sentence_uncertainties):
        sentences = [s for s in report.split(". ") if s.strip()]
        if not sentences:
            pruned.append(report)
            continue
        n_keep = max(1, len(sentences) - 1)
        kept = rng.sample(sentences, min(n_keep, len(sentences)))
        pruned.append(". ".join(kept))
    return pruned
