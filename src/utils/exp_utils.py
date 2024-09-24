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


def compute_rouge(predictions: List[str], references: List[str], tokenizer) -> Tuple[dict, List[str]]:
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    tokenizer = Tokenizer(tokenizer)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False, tokenizer=tokenizer)
    aggregator = scoring.BootstrapAggregator()

    new_predictions = []
    for ref, preds in zip(references, predictions):
        scores = [scorer.score(ref, pred) for pred in preds]
        idx = np.argmax([score["rouge2"].recall for score in scores])
        new_predictions.append(preds[idx])
        aggregator.add_scores(scores[idx])

    result = aggregator.aggregate()
    for key in result:
        result[key] = result[key].mid.recall

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
