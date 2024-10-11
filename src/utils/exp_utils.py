import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rouge_score import rouge_scorer, scoring
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter


class Tokenizer:
    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer_func(text)


class Accuracy:
    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def __call__(self, pred: str, ref: str):
        ref_ = []
        ref_.extend(list(re.findall(r"\[(\d), (\d)\]", ref)[0]))
        ref_.append(re.findall(r"shows a (\w+), while", ref)[0])
        ref_.append(re.findall(r"show a (\w+)\.", ref)[0])

        pred_ = []
        if re.findall(r"\[(\d), (\d)\]", pred) == []:
            pred_ = ["0", "0"]
        else:
            pred_.extend(list(re.findall(r"\[(\d), (\d)\]", pred)[0]))
        if re.findall(r"shows a (\w+), while", pred) == []:
            pred_.append("none")
        else:
            pred_.append(re.findall(r"shows a (\w+), while", pred)[0])
        if re.findall(r"show a (\w+)\.", pred) == []:
            pred_.append("none")
        else:
            pred_.append(re.findall(r"show a (\w+)\.", pred)[0])

        acc = 0
        for r, p in zip(ref_, pred_):
            if r == p:
                acc += 1

        return acc / 4


def compute_score(predictions: List[str], references: List[str], tokenizer) -> Tuple[dict, List[str]]:
    rouge_types = ["rouge1", "rouge2"]
    tokenizer = Tokenizer(tokenizer)
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False, tokenizer=tokenizer)
    aggregator = scoring.BootstrapAggregator()

    new_predictions = []
    for ref, preds in zip(references, predictions):
        scores = [scorer.score(ref, pred) for pred in preds]
        idx = np.argmax([score["rouge2"].fmeasure for score in scores])
        new_predictions.append(preds[idx])
        aggregator.add_scores(scores[idx])

    result = aggregator.aggregate()
    for key in result:
        result[key] = result[key].mid.fmeasure

    return result, new_predictions


class CustomLoss:
    def __init__(self, loss_weight: Tensor, writer: SummaryWriter):
        super(CustomLoss, self).__init__()
        self.lm_loss_func = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=-100)
        self.writer = writer

    def __call__(self, logits: Tensor, labels: Tensor, coords_labels, step: int) -> Tensor:
        lm_loss = self.lm_loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.writer.add_scalar("Loss/lm_loss", lm_loss.item(), step)
        return lm_loss
