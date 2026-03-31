# Reproducing Results

All experiments use MIMIC-CXR with RaDialog-generated reports (temperature=1, 10 samples per image).  
Data preparation follows [RaDialog](https://github.com/ChantalMP/RaDialog).

## Input Data Format

| Argument | Format |
|----------|--------|
| `--chexpert_file_path` | `.pkl` with keys `greedy_reports` (list of N strings) and `sampled_reports` (list of N × 10 strings) |
| `--predictions_file_path` | `.csv` with no header, first column = greedy-decoded report string |
| `--sampled_reports_path` | `.pkl` containing a list of N lists, each with 10 sampled report strings |
| `--green_scores_path` | `.pkl` with key `greens` (list of N tensors), or `.csv` with one value per row |
| `--green_uncertainty_path` | `.pkl` with key `uncertainty` (list of N tensors), or `.csv` with one value per row |
| `--u_nll_path` | `.csv` with column `u_nll` |
| `--u_normnll_path` | `.csv` with column `u_normnll` |
| `--u_lexicalsim_path` | `.csv` with column `ROUGE_L_UQ` |

## 1. Generate UQ Scores

```bash
# Report-level (VRO-GREEN)
python src/uq/VroGreen.py \
  --exp_name chexpert-plus \
  --chexpert_file_path <path/to/chexpert.pkl> \
  --output_base_path results \
  --num_samples 3858 --batch_size 16

# Sentence-level (VRO-RadGraph)
python src/uq/VroRadSent.py \
  --exp CheXpertPlus_mimiccxr \
  --chexpert_file <path/to/chexpert.pkl> \
  --num_samples 3858 --output_dir results/exp_result
```

## 2. Alignment (Table 1)

Pearson correlation between UQ estimates and oracle factuality metrics.

```bash
python src/alignment/report_alignment.py --exp RaDialog
python src/alignment/sentence_alignment.py --exp RaDialog
```

## 3. Abstention (Table 2 / Figure 2)

```bash
python src/abstention/report_abstention.py \
  --exp ChexpertPlus \
  --green_scores_path <path/to/green_scores.pkl> \
  --green_uncertainty_path <path/to/green_uncertainty.csv> \
  --u_lexicalsim_path <path/to/lexicalUQ.csv> \
  --output_base_path results

python src/abstention/sent_abstention.py --exp RaDialog
```

## 4. Hallucination / Prior-Reference Detection (Figure 3)

```bash
python src/hallucination/hallucination.py --exp RaDialog
```

## 5. Rank Calibration Error

```bash
python src/misc/cal_rce.py \
  --scores_path <path/to/green_scores.pkl> \
  --green_uncertainty_path <path/to/green_uncertainty.pkl> \
  --u_nll_path <path/to/u_nll.csv> \
  --u_lexicalsim_path <path/to/lexicalUQ.csv>
```

Run any script with `--help` to see all available options.
