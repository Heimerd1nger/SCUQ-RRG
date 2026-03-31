"""Sentence-level uncertainty quantification via RadGraph entity consistency."""

from dataclasses import dataclass, field
from typing import List
import math

import pandas as pd

from ..utils import ReportSample


@dataclass
class SentenceUncertaintyResult:
    """Per-report sentence-level UQ output."""
    sentences: List[str]
    uncertainty_scores: List[float]       # per-sentence score ∈ [0, 1]
    highest_uncertainty_idx: int          # index of most uncertain sentence
    flagged_sentence: str                 # text of most uncertain sentence


def count_consistent_entities_batch(hypothesis_annotation_lists, reference_annotation_lists):
    """Compute entity-level consistency ratios for a batch of (hyp, ref) annotation pairs."""
    consistent_ratios = []
    for hyp_ann, ref_ann in zip(hypothesis_annotation_lists, reference_annotation_lists):
        hyp_entities = hyp_ann["entities"]
        ref_entities = ref_ann["entities"]

        reference_set = [(e["tokens"], e["label"]) for e in ref_entities.values()]
        hypothesis_set = [(e["tokens"], e["label"]) for e in hyp_entities.values()]

        consistent_count = sum(
            1 for e in hyp_entities.values()
            if (e["tokens"], e["label"]) in reference_set
        )

        if hypothesis_set:
            ratio = consistent_count / len(hypothesis_set)
        else:
            ratio = float("nan")
        consistent_ratios.append(ratio)
    return consistent_ratios


class SentenceUncertaintyScorer:
    """Sentence-level UQ using RadGraph entity consistency (VRO-RadGraph).

    For each sentence in the original report, measures how consistently its
    entities appear across the stochastic samples. Higher score = more uncertain.

    Example
    -------
    >>> from scuq import ReportSample
    >>> from scuq.uq import SentenceUncertaintyScorer
    >>> scorer = SentenceUncertaintyScorer()
    >>> result = scorer.score(sample)
    >>> print(result.flagged_sentence)
    """

    def __init__(self, method: str = "vro_radgraph"):
        if method != "vro_radgraph":
            raise ValueError(f"Unknown method '{method}'. Currently supported: 'vro_radgraph'.")
        self.method = method
        self._f1radgraph = None  # lazy-load to avoid import cost at import time

    def _get_radgraph(self):
        if self._f1radgraph is None:
            try:
                from radgraph import F1RadGraph
            except ImportError:
                raise ImportError(
                    "radgraph is required for VRO-RadGraph. Install it with:\n"
                    "    pip install -e third_party/CXR-Report-Metric/"
                )
            self._f1radgraph = F1RadGraph(reward_level="complete")
        return self._f1radgraph

    def score(self, sample: ReportSample) -> SentenceUncertaintyResult:
        """Score all sentences in one report."""
        f1radgraph = self._get_radgraph()
        sentences = [s for s in sample.original_report.split(". ") if s.strip()]
        sentence_uncertainties = []

        for sent in sentences:
            batch_sents = [sent] * len(sample.sampled_reports)
            _, _, hyp_anns, ref_anns = f1radgraph(
                hyps=batch_sents,
                refs=sample.sampled_reports,
            )
            ratios = count_consistent_entities_batch(hyp_anns, ref_anns)
            # Mean consistency → uncertainty = 1 - consistency (ignore NaN)
            valid = [r for r in ratios if not math.isnan(r)]
            consistency = sum(valid) / len(valid) if valid else 0.0
            sentence_uncertainties.append(1.0 - consistency)

        if sentence_uncertainties:
            highest_idx = int(max(range(len(sentence_uncertainties)),
                                  key=lambda i: sentence_uncertainties[i]))
        else:
            highest_idx = 0

        return SentenceUncertaintyResult(
            sentences=sentences,
            uncertainty_scores=sentence_uncertainties,
            highest_uncertainty_idx=highest_idx,
            flagged_sentence=sentences[highest_idx] if sentences else "",
        )

    def score_batch(self, samples: List[ReportSample]) -> List[SentenceUncertaintyResult]:
        """Score a list of samples."""
        return [self.score(s) for s in samples]
