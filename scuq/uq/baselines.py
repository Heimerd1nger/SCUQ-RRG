"""Baseline UQ methods: lexical similarity (ROUGE-L).

Predictive entropy (PE) and length-normalised PE are computed from model
log-probabilities during sampling and stored as pre-computed CSV columns
(``u_nll``, ``u_normnll``). Load them directly with pandas; no scorer class
is needed here.
"""

import numpy as np
from typing import List

from ..utils import ReportSample


def _preprocess_text(s: str) -> str:
    from nltk import word_tokenize
    s = s.replace("\n", "").replace("<s>", "").replace("</s>", "")
    return " ".join(word_tokenize(s.lower()))


class LexicalSimilarityScorer:
    """Lexical similarity baseline using ROUGE-L.

    Uncertainty = 1 - mean_ROUGE_L(original_report, sampled_reports).

    Requires ``pycocoevalcap``:
        pip install pycocoevalcap

    Example
    -------
    >>> scorer = LexicalSimilarityScorer()
    >>> uncertainty = scorer.score(sample)  # float in [0, 1]
    """

    def __init__(self):
        from pycocoevalcap.rouge.rouge import Rouge
        self._rouge = Rouge()

    def score(self, sample: ReportSample) -> float:
        """Returns lexical-similarity uncertainty ∈ [0, 1]."""
        original = _preprocess_text(sample.original_report)
        rouge_scores = []
        for sampled in sample.sampled_reports:
            s = _preprocess_text(sampled)
            gts = {0: [original]}
            res = {0: [s]}
            rouge_l, _ = self._rouge.compute_score(gts, res)
            rouge_scores.append(rouge_l)
        return float(1.0 - np.mean(rouge_scores))

    def score_batch(self, samples: List[ReportSample]) -> List[float]:
        return [self.score(s) for s in samples]
