"""Command-line tool to compute Rank Calibration Error (RCE) and related metrics."""

import argparse
import pickle
import numpy as np
import pandas as pd

from scuq.calibration import evaluate_calibration


def load_scores(path):
    """Load GREEN scores from pickle (tensor list) or CSV."""
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            scores = data.get('greens', data.get('uncertainty', list(data.values())[0]))
        else:
            scores = data
        return np.array([float(t) if not hasattr(t, 'numpy') else t.numpy() for t in scores])
    else:
        df = pd.read_csv(path, header=None)
        return np.array([float(str(v).replace("tensor(", "").replace(")", "")) for v in df[0].values])


def load_uncertainty(path, col=None):
    """Load uncertainty scores from CSV or pickle."""
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            arr = data.get('uncertainty', list(data.values())[0])
        else:
            arr = data
        return np.array([float(t) if not hasattr(t, 'numpy') else t.numpy() for t in arr])
    else:
        df = pd.read_csv(path)
        if col and col in df.columns:
            return df[col].values
        return df.iloc[:, 0].values


def main(args):
    scores = load_scores(args.scores_path)

    methods = {}
    if args.green_uncertainty_path:
        methods['VRO-GREEN'] = load_uncertainty(args.green_uncertainty_path)
    if args.u_nll_path:
        methods['Predictive entropy'] = load_uncertainty(args.u_nll_path, col='u_nll')
    if args.u_normnll_path:
        methods['Normalised entropy'] = load_uncertainty(args.u_normnll_path, col='u_normnll')
    if args.u_lexicalsim_path:
        raw = load_uncertainty(args.u_lexicalsim_path, col='ROUGE_L_UQ')
        methods['Lexical similarity'] = 1 - raw

    if not methods:
        print("No uncertainty files provided. Use --green_uncertainty_path, --u_nll_path, etc.")
        return

    header = f"{'Method':<25} {'RCE':>8} {'Pearson r':>10} {'Spearman r':>11} {'AUROC':>8}"
    print(header)
    print("-" * len(header))

    for name, uncertainty in methods.items():
        n = min(len(scores), len(uncertainty))
        result = evaluate_calibration(uncertainty[:n], scores[:n])
        print(f"{name:<25} {result['rce']:>8.4f} {result['pearson_r']:>10.4f} "
              f"{result['spearman_r']:>11.4f} {result['auroc']:>8.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute RCE and calibration metrics.')
    parser.add_argument('--scores_path', type=str, required=True,
                        help='Path to factuality scores (pkl or csv)')
    parser.add_argument('--green_uncertainty_path', type=str, default=None,
                        help='Path to VRO-GREEN uncertainty (pkl or csv)')
    parser.add_argument('--u_nll_path', type=str, default=None,
                        help='Path to predictive entropy CSV (column: u_nll)')
    parser.add_argument('--u_normnll_path', type=str, default=None,
                        help='Path to normalised entropy CSV (column: u_normnll)')
    parser.add_argument('--u_lexicalsim_path', type=str, default=None,
                        help='Path to lexical similarity CSV (column: ROUGE_L_UQ)')
    args = parser.parse_args()
    main(args)
