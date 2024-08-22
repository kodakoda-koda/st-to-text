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

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    for key in result:
        result[key] = result[key].mid.recall

    return result
