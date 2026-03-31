# Reproducing Results

All experiments use MIMIC-CXR with RaDialog-generated reports (temperature=1, 10 samples per image).  
Data preparation follows [RaDialog](https://github.com/ChantalMP/RaDialog).

## 1. Generate UQ Scores

```bash
# Report-level (VRO-GREEN)
python src/uq/VroGreen.py --exp_name chexpert-plus --num_samples 3858

# Sentence-level (VRO-RadGraph)
python src/uq/VroRadSent.py --exp CheXpertPlus_mimiccxr --num_samples 3858
```

## 2. Alignment (Table 1)

Pearson correlation between UQ estimates and oracle factuality metrics.

```bash
python src/alignment/report_alignment.py --exp RaDialog
python src/alignment/sentence_alignment.py --exp RaDialog
```

## 3. Abstention (Table 2 / Figure 2)

```bash
python src/abstention/report_abstention.py --exp RaDialog
python src/abstention/sent_abstention.py --exp RaDialog
```

## 4. Hallucination / Prior-Reference Detection (Figure 3)

```bash
python src/hallucination/hallucination.py --exp RaDialog
```

## 5. Rank Calibration Error

```bash
python src/misc/cal_rce.py
```

Run any script with `--help` to see all available options and default data paths.
