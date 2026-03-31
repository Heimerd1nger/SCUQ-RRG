from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ReportSample:
    """One X-ray study: original report + stochastic samples.

    Args:
        study_id: Identifier used for result tracking.
        original_report: The report to score (typically greedy-decoded).
        sampled_reports: T stochastic samples from the same model (T >= 5 recommended).
        reference_report: Ground-truth report; required for factuality scoring.
    """
    study_id: str
    original_report: str
    sampled_reports: List[str]
    reference_report: Optional[str] = None


def calculate_accuracy_and_improvement(
    uncertainty_scores: np.ndarray,
    factual_scores: np.ndarray,
) -> tuple:
    """Compute mean factuality at each abstention threshold (5 % steps).

    Returns:
        all_acc: list of mean factuality scores at each threshold.
        improvements: relative improvement over the no-abstention baseline.
    """
    all_acc = []
    sorted_indices = np.argsort(uncertainty_scores)

    for i in range(5, 105, 5):
        num = max(1, int(len(uncertainty_scores) * i / 100))
        top_indices = sorted_indices[0:num]
        acc = factual_scores[top_indices].mean()
        all_acc.append(acc)

    improvements = [(acc - all_acc[-1]) / all_acc[-1] for acc in all_acc]
    return all_acc, improvements
