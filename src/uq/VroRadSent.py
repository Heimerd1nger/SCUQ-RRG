"""Command-line interface for sentence-level VRO-RadGraph uncertainty quantification."""

import argparse
import pickle

import pandas as pd

from scuq.uq.sentence_uq import count_consistent_entities_batch


def remove_ith_sentence(samples, i):
    updated_samples = []
    for sample in samples:
        sentences = sample.split('. ')
        if 0 <= i < len(sentences):
            del sentences[i]
        updated_sample = '. '.join(sentences)
        updated_samples.append(updated_sample)
    return updated_samples


def process_samples(args):
    print(args.exp)

    if args.exp == 'CheXpertPlus_mimiccxr':
        with open(args.chexpert_file, 'rb') as file:
            chexpert_plus = pickle.load(file)
        greedy = chexpert_plus['greedy_reports']
        all_sample_list = [list(item) for item in chexpert_plus['sampled_reports']]
    else:
        with open(args.sampled_reports_file, 'rb') as file:
            all_sample_list = pickle.load(file)
        all_pred = pd.read_csv(args.predictions_file, header=None)
        all_pred_list = all_pred.values.tolist()
        greedy = [item[0] for item in all_pred_list]

    try:
        from radgraph import F1RadGraph
    except ImportError:
        raise ImportError(
            "radgraph is required for VRO-RadGraph. Install it with:\n"
            "    pip install radgraph"
        )

    mode = "complete"
    f1radgraph = F1RadGraph(reward_level=mode)
    all_cent = []

    for i in range(args.num_samples):
        print(i)
        samples = greedy[i]
        u_cent = []
        for sent in samples.split('. '):
            batch_sents = [sent] * 10
            mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
                hyps=batch_sents, refs=all_sample_list[i]
            )
            tmp = count_consistent_entities_batch(hypothesis_annotation_lists, reference_annotation_lists)
            u_cent.append(tmp)
        all_cent.append(u_cent)

    output_file = f"{args.output_dir}/{args.exp}-rad_cent_consistent-{args.num_samples}.pkl"
    with open(output_file, 'wb') as file:
        pickle.dump(all_cent, file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some samples.")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--exp", type=str, default="CheXpertPlus_mimiccxr")
    parser.add_argument("--chexpert_file", type=str, default="data/batch_chexpert_mimix_cxr_num3858.pkl")
    parser.add_argument("--sampled_reports_file", type=str, default="data/sampled_reports_num_beams1.pkl")
    parser.add_argument("--predictions_file", type=str,
                        default="data/predictions_checkpoints_vicuna-7b-img-report_checkpoint-11200.csv")
    parser.add_argument("--output_dir", type=str, default="results/exp_result")

    args = parser.parse_args()
    process_samples(args)
