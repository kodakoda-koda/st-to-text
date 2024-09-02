import numpy as np
import torch
import torch.nn as nn
from rouge_score import rouge_scorer, scoring


class Tokenizer:
    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


def compute_rouge(predictions, references, tokenizer):
    rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    tokenizer = Tokenizer(tokenizer)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=False, tokenizer=tokenizer)
    aggregator = scoring.BootstrapAggregator()

    new_predictions = []
    for ref, preds in zip(references, predictions):
        scores = [scorer.score(ref, pred) for pred in preds]
        idx = np.argmax([score["rouge1"].recall for score in scores])
        new_predictions.append(preds[idx])
        aggregator.add_scores(scores[idx])

    result = aggregator.aggregate()
    for key in result:
        result[key] = result[key].mid.recall

    return result, new_predictions


class CustomLoss:
    def __init__(self, loss_weight):
        super(CustomLoss, self).__init__()
        self.lm_loss_func = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=-100)
        self.x_idx = [4347, 4482, 6355, 8525, 11116, 11071, 32100, 11864]
        self.y_idx = [209, 204, 220, 314, 305, 431, 489, 505]
        self.softmax = nn.Softmax(dim=-1)
        self.dist = None

    def __call__(self, logits, labels):
        x_labels = labels[:, 2].to(torch.device("cpu")).apply_(lambda x: self.x_idx.index(x) + 1).to(logits.device)
        y_labels = labels[:, 3].to(torch.device("cpu")).apply_(lambda x: self.y_idx.index(x) + 1).to(logits.device)

        x_prob = self.softmax(logits[:, 2])
        y_prob = self.softmax(logits[:, 3])

        x_coords = torch.argmax(x_prob, dim=-1).to(torch.device("cpu")).apply_(self.__get_coord__).to(logits.device)
        y_coords = torch.argmax(y_prob, dim=-1).to(torch.device("cpu")).apply_(self.__get_coord__).to(logits.device)
        x_dist = torch.abs(x_coords - x_labels)
        y_dist = torch.abs(y_coords - y_labels)
        self.dist = x_dist + y_dist

        x_loss = x_prob.max(dim=-1)[0] * x_dist
        y_loss = y_prob.max(dim=-1)[0] * y_dist

        lm_loss = self.lm_loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        coord_loss = x_loss.mean() + y_loss.mean()
        return lm_loss + coord_loss * 0.2

    def __get_coord__(self, i):
        if i not in self.x_idx + self.y_idx:
            return 100
        elif i in self.x_idx:
            return self.x_idx.index(i) + 1
        else:
            return self.y_idx.index(i) + 1

    def __get_dist__(self):
        return self.dist
