"""Command-line interface for report-level VRO-GREEN uncertainty quantification.

Core GREENModel/GREEN classes live in scuq.uq.report_uq.
This script adds batch-processing logic and CLI argument parsing.
"""

import argparse
import pickle
import time

import pandas as pd

from scuq.uq.report_uq import GREEN


def process_batch(model, refs, hyps):
    mean_green, greens, text = model(refs=refs, hyps=hyps)
    return mean_green, greens, text


def main(args):
    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
        do_sample=False,
        batch_size=args.batch_size,
        return_0_if_no_green_score=True,
        cuda=True,
    )

    if args.exp_name == 'chexpert-plus':
        with open(args.chexpert_file_path, 'rb') as file:
            all_ = pickle.load(file)
        refs = all_['greedy_reports']
        hyps = [list(item) for item in all_['sampled_reports']]
    else:
        all_pred = pd.read_csv(args.predictions_file_path, header=None)
        all_pred_list = all_pred.values.tolist()

        with open(args.sampled_reports_path, 'rb') as file:
            all_sample_list = pickle.load(file)

        refs = [item[0] for item in all_pred_list]
        hyps = all_sample_list

    all_greens = []
    all_text = []
    all_green_uncertainty = []

    for i in range(args.num_samples):
        mean_green, greens, text = process_batch(model, [refs[i]] * len(hyps[i]), hyps[i])
        print(f'Green uncertainty for {i}-th sample is {1 - mean_green}')
        all_greens.extend(greens)
        all_text.extend(text)
        all_green_uncertainty.append(1 - mean_green.item())

    output_path = f'{args.output_base_path}/{args.exp_name}/green_uncertainty-{args.num_samples}.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump({'greens': all_greens, 'text': all_text, 'uncertainty': all_green_uncertainty}, file)
    print(f"Results saved to {output_path}")

    greens_csv_path = f'{args.output_base_path}/{args.exp_name}/green_uncertainty-{args.num_samples}.csv'
    greens_df = pd.DataFrame(all_green_uncertainty, columns=['Green Uncertainty'])
    greens_df.to_csv(greens_csv_path, index=False, header=False)
    print(f"Greens uncertainty saved to {greens_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate GREEN scores for a given number of samples.')
    parser.add_argument('--num_samples', type=int, default=3858)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--exp_name', type=str, default='chexpert-plus')
    parser.add_argument('--chexpert_file_path', type=str, default='data/batch_chexpert_mimix_cxr_num3858.pkl')
    parser.add_argument('--sampled_reports_path', type=str, default='data/sampled_reports_num_beams1.pkl')
    parser.add_argument('--predictions_file_path', type=str,
                        default='data/predictions_checkpoints_vicuna-7b-img-report_checkpoint-11200.csv')
    parser.add_argument('--output_base_path', type=str, default='results')
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print("Total time taken: ", time.time() - start_time)
