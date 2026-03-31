# SCUQ-RRG

[![Paper](https://img.shields.io/badge/NAACL%202025-Findings-red)](https://aclanthology.org/2025.findings-naacl.95.pdf)
[![Citations](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2FACL%3A2025.findings-naacl.95%3Ffields%3DcitationCount&query=%24.citationCount&label=Cited%20by&color=blue)](https://aclanthology.org/2025.findings-naacl.95/)
[![PyPI](https://img.shields.io/pypi/v/scuq-rrg?color=orange)](https://pypi.org/project/scuq-rrg/)
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

See [`example/example_data.ipynb`](example/example_data.ipynb) for the exact pickle/CSV format used in experiments.

## Demos

- [`example/VRO_GREEN_demo.ipynb`](example/VRO_GREEN_demo.ipynb) — report-level UQ walkthrough
- [`example/VRO_Rad_demo.ipynb`](example/VRO_Rad_demo.ipynb) — sentence-level UQ walkthrough
- [`example/quickstart.ipynb`](example/quickstart.ipynb) — end-to-end quickstart

## Running Experiments

### Report Scores

```bash
python -m src.uq.VroGreen \
  --exp_name chexpert-plus \
  --chexpert_file_path data/batch_chexpert_mimix_cxr_num3858.pkl \
  --output_base_path results \
  --num_samples 3858 --batch_size 16
```

### Sentence UQ

```bash
python -m src.uq.VroRadSent \
  --exp CheXpertPlus_mimiccxr \
  --chexpert_file data/batch_chexpert_mimix_cxr_num3858.pkl \
  --num_samples 3858 --output_dir results/exp_result
```

### Abstention

```bash
python src/abstention/report_abstention.py \
  --exp ChexpertPlus \
  --green_scores_path data/green_scores-3858.pkl \
  --green_uncertainty_path results/chexpert-plus/green_uncertainty-3858.csv \
  --u_lexicalsim_path data/uq/lexicalUQ.csv \
  --output_base_path results
```

### Calibration (RCE)

```bash
python src/misc/cal_rce.py \
  --scores_path data/green_scores-3858.pkl \
  --green_uncertainty_path results/chexpert-plus/green_uncertainty-3858.pkl \
  --u_nll_path data/uq/u_nll.csv \
  --u_lexicalsim_path data/uq/lexicalUQ.csv
```

## Citation

```bibtex
@inproceedings{wang2025semantic,
  title={Semantic Consistency-Based Uncertainty Quantification for Factuality in Radiology Report Generation},
  author={Wang, Chenyu and Bhatt, Parth and Shrivastava, Harshit and Bittencourt, Lucas and Kalra, Mannudeep K. and Gichoya, Judy W. and Celi, Leo Anthony and Peng, Yuyin and Patel, Bhavik N. and Trivedi, Hari},
  booktitle={Proceedings of the 2025 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2025}
}
```
