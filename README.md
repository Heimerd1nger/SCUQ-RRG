# SCUQ-RRG

[![arXiv](https://img.shields.io/badge/arXiv-2412.04606-b31b1b)](https://arxiv.org/abs/2412.04606)
[![Citations](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2FarXiv%3A2412.04606%3Ffields%3DcitationCount&query=%24.citationCount&label=Cited%20by&color=blue)](https://aclanthology.org/2025.findings-naacl.95/)
[![PyPI](https://img.shields.io/pypi/v/scuq-rrg?color=orange&cacheSeconds=3600)](https://pypi.org/project/scuq-rrg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Code for the NAACL 2025 paper **"Semantic Consistency-Based Uncertainty Quantification for Factuality in Radiology Report Generation"**.

## Install

```bash
pip install scuq-rrg
```

For full functionality (GREEN model and RadGraph):

```bash
git clone --recurse-submodules https://github.com/Heimerd1nger/SCUQ-RRG.git
cd SCUQ-RRG
pip install -e .
pip install -e third_party/GREEN/   # green_score (PyPI version has incompatible API)
pip install radgraph                # sentence-level UQ
```

## Usage

### Report-Level Uncertainty (VRO-GREEN)

Measures report-level factual uncertainty by comparing a greedy-decoded report against multiple sampled reports using the GREEN metric.

```python
from scuq import ReportUncertaintyScorer

scorer = ReportUncertaintyScorer(
    model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
    cuda=True,
)

# greedy_report: the reference (greedy-decoded) report
# sampled_reports: list of 10 stochastically sampled reports
greedy_report = "The lungs are clear. No pleural effusion. Cardiomediastinal silhouette is normal."
sampled_reports = [
    "Lungs are clear bilaterally. No effusion or pneumothorax.",
    "Clear lungs. Heart size normal. No acute findings.",
    # ... (typically 10 samples)
]

result = scorer.score(greedy_report, sampled_reports)
print(f"Uncertainty: {result.uncertainty:.3f}")   # e.g. 0.596
print(f"Mean GREEN:  {result.mean_green:.3f}")    # e.g. 0.404
```

### Sentence-Level Uncertainty (VRO-RadGraph)

Identifies the most uncertain sentence in a report using RadGraph entity consistency.

```python
from scuq import SentenceUncertaintyScorer

scorer = SentenceUncertaintyScorer()

greedy_report = (
    "No pneumothorax. "
    "Possible left lower lobe opacity suggesting pneumonia. "
    "Mild cardiomegaly. "
    "No pleural effusion. "
    "Stable appearance compared to prior. "
    "No acute osseous abnormality."
)
sampled_reports = [
    "No pneumothorax or effusion. Heart size normal.",
    "Bilateral lungs clear. No acute findings.",
    # ...
]

result = scorer.score(greedy_report, sampled_reports)
# Per-sentence uncertainty scores (0 = certain, 1 = uncertain):
# [0.05, 0.60, 0.80, 0.40, 0.28, 0.10]
print(f"Most uncertain: '{result.flagged_sentence}'")
print(f"Sentence scores: {[round(s, 2) for s in result.uncertainty_scores]}")
```

## Data Format

Experiments expect:
- **`greedy_reports`**: list of N strings (greedy-decoded reports)
- **`sampled_reports`**: list of N lists, each with 10 sampled strings

## Citation

```bibtex
@inproceedings{wang2025semantic,
  title={Semantic consistency-based uncertainty quantification for factuality in radiology report generation},
  author={Wang, Chenyu and Zhou, Weichao and Ghosh, Shantanu and Batmanghelich, Kayhan and Li, Wenchao},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={1739--1754},
  year={2025}
}
```
