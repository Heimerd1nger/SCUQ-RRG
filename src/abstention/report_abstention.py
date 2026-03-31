import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
import os
import argparse
from scipy.stats import pearsonr

from scuq.utils import calculate_accuracy_and_improvement


def main(args):
    if args.exp == "RaDialog":
        plot_radialog(args)
    elif args.exp == "ChexpertPlus":
        plot_chexpertplus(args)
    else:
        raise ValueError("Invalid experiment name. Use 'RaDialog' or 'ChexpertPlus'.")


def plot_radialog(args):
    with open(args.green_scores_path, 'rb') as file:
        green_score = pickle.load(file)
    us = np.array([t.numpy() for t in green_score['greens']])

    u_nll = pd.read_csv(args.u_nll_path)['u_nll'].values
    u_normnll = pd.read_csv(args.u_normnll_path)['u_normnll'].values

    with open(args.green_uncertainty_path, 'rb') as file:
        green_uncertainty = pickle.load(file)
    ugreen = np.array([t.numpy() for t in green_uncertainty['uncertainty']])

    u_lexicalsim = pd.read_csv(args.u_lexicalsim_path)['ROUGE_L_UQ'].values
    u_lexicalsim = [1 - i for i in u_lexicalsim]

    uncertainties = [u_nll, u_normnll, ugreen, u_lexicalsim]
    labels = ['Predictive entropy', 'Normalised entropy', 'VRO-GREEN(Ours)', 'Lexical similarity']
    colors = ['b', 'g', 'r', 'm']

    plot_factuality_scores(uncertainties, labels, colors, us,
                           "Effect of Report Abstention on Factuality Scores across Different UQ Methods", args)


def plot_chexpertplus(args):
    score = pd.read_csv(args.green_scores_path, header=None)
    score = np.array([float(t.replace("tensor(", "").replace(")", "")) for t in score[0].values])

    ugreen = pd.read_csv(args.green_uncertainty_path, header=None)
    ugreen = np.array([float(t.replace("tensor(", "").replace(")", "")) for t in ugreen[0].values])

    u_lexicalsim = pd.read_csv(args.u_lexicalsim_path)['ROUGE_L_UQ'].values
    uncertainties = [ugreen, -u_lexicalsim]
    labels = ['VRO-GREEN(Ours)', 'Lexical similarity']
    colors = ['r', 'm']

    plot_factuality_scores(uncertainties, labels, colors, score,
                           "Effect of Report Abstention on Factuality Scores across Different UQ Methods", args)


def plot_factuality_scores(uncertainties, labels, colors, scores, title, args):
    all_annotations = {}
    for uncertainty, label, color in zip(uncertainties, labels, colors):
        all_acc, improvements = calculate_accuracy_and_improvement(uncertainty, scores)
        for x, y, imp in zip(range(95, -5, -5), all_acc, improvements):
            if x % 20 == 0:
                if x not in all_annotations:
                    all_annotations[x] = []
                all_annotations[x].append((y, imp, color))

    plt.figure(figsize=(7, 3.5), dpi=150)

    for uncertainty, label, color in zip(uncertainties, labels, colors):
        all_acc, improvements = calculate_accuracy_and_improvement(uncertainty, scores)
        plt.plot(range(95, -5, -5), all_acc, marker='o', linestyle='-', linewidth=1.5, color=color,
                 label=f'Factuality Score ({label})')

    for x, y_imp_pairs in all_annotations.items():
        top_2_pairs = sorted(y_imp_pairs, key=lambda pair: pair[1], reverse=True)[:2]
        for y, imp, color in top_2_pairs:
            plt.text(x, y, f'{imp:.2%}', fontsize=8, ha='right', color=color,
                     bbox=dict(facecolor='white', alpha=0.5))

    plt.title(title, fontsize=14)
    plt.xlabel('Percentage of Reports Abstained (%)', fontsize=14)
    plt.ylabel('Factuality Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.6, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    output_path = os.path.join(args.output_base_path, f'{args.exp}_report_abstention.png')
    plt.savefig(output_path, format='png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Factuality Scores for Different UQ Methods")
    parser.add_argument('--exp', type=str, required=True, help="Experiment name: 'RaDialog' or 'ChexpertPlus'")
    parser.add_argument('--green_scores_path', type=str, default='data/green_scores-3858.pkl')
    parser.add_argument('--u_nll_path', type=str, default='data/uq/u_nll.csv')
    parser.add_argument('--u_normnll_path', type=str, default='data/uq/u_normnll.csv')
    parser.add_argument('--green_uncertainty_path', type=str, default='data/green_uncertainty-3858.pkl')
    parser.add_argument('--u_lexicalsim_path', type=str, default='data/uq/lexicalUQ.csv')
    parser.add_argument('--output_base_path', type=str, default='results')
    args = parser.parse_args()
    main(args)
