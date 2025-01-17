import re
from typing import List, Tuple

import evaluate
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


def compute_score(predictions: List[str], references: List[str], tokenizer):
    predictions = aggregate_prediction(predictions, references, tokenizer)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    rouge_score = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2"],
        tokenizer=tokenizer,
    )
    bleu_score = bleu.compute(
        predictions=predictions,
        references=references,
        tokenizer=tokenizer,
    )

    scores = {
        "rouge1": rouge_score["rouge1"],
        "rouge2": rouge_score["rouge2"],
        "bleu": bleu_score["bleu"],
    }

    return scores, predictions


def aggregate_prediction(predictions: List[str], references: List[str], tokenizer) -> List[str]:
    rouge_types = ["rouge2"]
    tokenizer = Tokenizer(tokenizer)
    rouge_scorer_ = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False, tokenizer=tokenizer)

    new_predictions = []
    for ref, preds in zip(references, predictions):
        scores = [rouge_scorer_.score(ref, pred)["rouge2"].fmeasure for pred in preds]
        idx = np.argmax(scores)
        new_predictions.append(preds[idx])

    return new_predictions


class CustomLoss:
    def __init__(self, loss_weight: Tensor, writer: SummaryWriter):
        super(CustomLoss, self).__init__()
        self.lm_loss_func = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=-100)
        self.writer = writer

    def __call__(self, logits: Tensor, labels: Tensor, step: int) -> Tensor:
        lm_loss = self.lm_loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.writer.add_scalar("Loss/lm_loss", lm_loss.item(), step)
        return lm_loss
