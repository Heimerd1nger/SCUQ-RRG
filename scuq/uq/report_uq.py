"""Report-level uncertainty quantification via VRO-GREEN.

Core model classes (GREENModel, GREEN) are preserved as-is from the original
implementation. ReportUncertaintyScorer provides the clean public API.
"""

import re
from typing import List

import torch
import torch.nn as nn
import pandas as pd

from ..utils import ReportSample


def _import_green_utils():
    """Lazy import of green_score.utils — only required when GREEN model is used."""
    try:
        from green_score.utils import process_responses, make_prompt, tokenize_batch_as_chat, truncate_to_max_len
        return process_responses, make_prompt, tokenize_batch_as_chat, truncate_to_max_len
    except ImportError:
        raise ImportError(
            "green_score is required for VRO-GREEN. Install it with:\n"
            "    pip install -e third_party/GREEN/"
        )

# Module-level cache: avoids recomputing GREEN scores for identical (ref, hyp) pairs
_pair_to_reward_dict: dict = {}


class GREENModel(nn.Module):
    """LLM-based radiology report evaluator wrapping StanfordAIMI/GREEN-radllama2-7b.

    Parses model outputs to extract clinically significant error counts and
    matched findings, then computes:
        score = matched_findings / (matched_findings + sig_clinical_errors)
    """

    def __init__(
        self,
        cuda: bool,
        model_id_or_path: str,
        do_sample: bool = False,
        batch_size: int = 4,
        return_0_if_no_green_score: bool = True,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cuda = cuda
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.return_0_if_no_green_score = return_0_if_no_green_score

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            trust_remote_code=True,
            device_map={"": "cuda:{}".format(torch.cuda.current_device())} if cuda else "cpu",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = (
            "{% for message in messages %}\n"
            "{% if message['from'] == 'human' %}\n"
            "{{ '<|user|>\n' + message['value'] + eos_token }}\n"
            "{% elif message['from'] == 'system' %}\n"
            "{{ '<|system|>\n' + message['value'] + eos_token }}\n"
            "{% elif message['from'] == 'gpt' %}\n"
            "{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n"
            "{% endif %}\n"
            "{% if loop.last and add_generation_prompt %}\n"
            "{{ '<|assistant|>' }}\n"
            "{% endif %}\n"
            "{% endfor %}"
        )

        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

    def get_response(self, input_ids, attention_mask):
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.do_sample,
            max_length=2048,
            temperature=None,
            top_p=None,
        )
        process_responses, _, _, _ = _import_green_utils()
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return process_responses(responses), outputs

    def parse_error_counts(self, text: str, category: str):
        if category not in self.categories:
            raise ValueError(f"Category {category} not in {self.categories}.")

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0] * 6

        if not category_text:
            if self.return_0_if_no_green_score:
                return sum_counts, sub_counts
            return None, [None] * 6

        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if counts:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts

        sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))
        if not matches:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            sub_categories = [f"({i}) " for i in range(1, len(self.sub_categories) + 1)]

        for position, sub_category in enumerate(sub_categories):
            for match in matches:
                if match.startswith(sub_category):
                    count = re.findall(r"(?<=: )\b\d+\b(?=\.)", match)
                    if count:
                        sub_counts[position] = int(count[0])
        return sum(sub_counts), sub_counts

    def compute_green(self, response: str):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0
        if sig_present is None or matched_findings is None:
            return None
        return matched_findings / (matched_findings + sum(sig_errors))

    def forward(self, input_ids, attention_mask):
        if self.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        reward_model_responses, output_ids = self.get_response(input_ids, attention_mask)
        greens = [self.compute_green(r) for r in reward_model_responses]
        greens = [g for g in greens if g is not None]
        return torch.tensor(greens, dtype=torch.float), output_ids


class GREEN(nn.Module):
    """Batching wrapper around GREENModel with pair-level caching."""

    def __init__(self, cuda: bool, max_len: int = 200, **kwargs):
        super().__init__()
        self.cuda = cuda
        self.max_len = max_len
        self.model = GREENModel(cuda, **kwargs)
        self.tokenizer = self.model.tokenizer
        if self.cuda:
            print("Using {} GPUs!".format(torch.cuda.device_count()))

    def forward(self, refs: List[str], hyps: List[str]):
        process_responses, make_prompt, tokenize_batch_as_chat, truncate_to_max_len = _import_green_utils()

        assert len(refs) == len(hyps)
        refs = truncate_to_max_len(refs, self.max_len)
        hyps = truncate_to_max_len(hyps, self.max_len)

        with torch.no_grad():
            pairs_to_process = []
            final_scores = torch.zeros(len(refs))
            output_ids_dict = {}

            for i, (ref, hyp) in enumerate(zip(refs, hyps)):
                if (ref, hyp) in _pair_to_reward_dict:
                    final_scores[i], output_ids_dict[i] = _pair_to_reward_dict[(ref, hyp)]
                else:
                    pairs_to_process.append((ref, hyp, i))

            if pairs_to_process:
                batch = [make_prompt(ref, hyp) for ref, hyp, _ in pairs_to_process]
                batch = [[{"from": "human", "value": p}, {"from": "gpt", "value": ""}] for p in batch]
                batch = tokenize_batch_as_chat(self.tokenizer, batch)

                greens_tensor, output_ids = self.model(batch["input_ids"], batch["attention_mask"])

                if len(greens_tensor) == len(pairs_to_process):
                    for (ref, hyp, idx), score, out_id in zip(pairs_to_process, greens_tensor, output_ids):
                        _pair_to_reward_dict[(ref, hyp)] = (score, out_id)
                        final_scores[idx] = score
                        output_ids_dict[idx] = out_id
                else:
                    print("Warning: inconsistency detected in pair processing.")

            responses = [output_ids_dict[i] for i in range(len(refs))]
            responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            mean_green = final_scores.mean()
            return mean_green, final_scores, process_responses(responses)


class ReportUncertaintyScorer:
    """Clean API for report-level uncertainty quantification.

    Supported methods
    -----------------
    ``"vro_green"``  — VRO with GREEN semantic parser (paper default, requires GPU)

    Example
    -------
    >>> from scuq import ReportSample, ReportUncertaintyScorer
    >>> scorer = ReportUncertaintyScorer(method="vro_green")
    >>> sample = ReportSample(
    ...     study_id="patient_001",
    ...     original_report="The lungs are clear.",
    ...     sampled_reports=["Lungs are clear.", "No consolidation.", ...],
    ... )
    >>> uncertainty = scorer.score(sample)  # float in [0, 1]
    """

    def __init__(
        self,
        method: str = "vro_green",
        model_id_or_path: str = "StanfordAIMI/GREEN-radllama2-7b",
        batch_size: int = 16,
        cuda: bool = True,
    ):
        if method != "vro_green":
            raise ValueError(f"Unknown method '{method}'. Currently supported: 'vro_green'.")
        self.method = method
        self._model = GREEN(
            cuda=cuda,
            model_id_or_path=model_id_or_path,
            do_sample=False,
            batch_size=batch_size,
            return_0_if_no_green_score=True,
        )

    def score(self, sample: ReportSample) -> float:
        """Score a single sample. Returns uncertainty ∈ [0, 1] (higher = more uncertain)."""
        refs = [sample.original_report] * len(sample.sampled_reports)
        mean_green, _, _ = self._model(refs=refs, hyps=sample.sampled_reports)
        return float(1 - mean_green.item())

    def score_batch(self, samples: List[ReportSample]) -> List[float]:
        """Score a list of samples."""
        return [self.score(s) for s in samples]

    def score_dataframe(
        self,
        df: pd.DataFrame,
        original_col: str = "original_report",
        sampled_col: str = "sampled_reports",
        id_col: str = "study_id",
    ) -> pd.DataFrame:
        """Process a DataFrame; returns df with an added ``uncertainty`` column."""
        import ast

        results = []
        for _, row in df.iterrows():
            sampled = row[sampled_col]
            if isinstance(sampled, str):
                sampled = ast.literal_eval(sampled)
            sample = ReportSample(
                study_id=str(row[id_col]),
                original_report=row[original_col],
                sampled_reports=sampled,
            )
            results.append(self.score(sample))

        out = df.copy()
        out["uncertainty"] = results
        return out
