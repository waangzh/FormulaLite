"""Small corpus metrics used by validation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def corpus_bleu(
    predictions: Sequence[str], references: Sequence[str], *, max_order: int = 4
) -> float:
    """Compute a corpus BLEU score in [0, 1] over normalized LaTeX tokens."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("at least one prediction/reference pair is required")

    clipped = [0] * max_order
    totals = [0] * max_order
    prediction_length = 0
    reference_length = 0
    for prediction, reference in zip(predictions, references, strict=True):
        prediction_tokens = prediction.split()
        reference_tokens = reference.split()
        prediction_length += len(prediction_tokens)
        reference_length += len(reference_tokens)
        for order in range(1, max_order + 1):
            prediction_counts = _ngrams(prediction_tokens, order)
            reference_counts = _ngrams(reference_tokens, order)
            clipped[order - 1] += sum(
                min(count, reference_counts[ngram])
                for ngram, count in prediction_counts.items()
            )
            totals[order - 1] += sum(prediction_counts.values())

    if prediction_length == 0:
        return 0.0
    precisions = [
        matches / total
        for matches, total in zip(clipped, totals, strict=True)
        if total > 0
    ]
    if not precisions or any(precision == 0 for precision in precisions):
        return 0.0
    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    mean_log_precision = sum(math.log(value) for value in precisions) / len(precisions)
    return brevity_penalty * math.exp(mean_log_precision)


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Return the dataset mean of token-level normalized Levenshtein distance."""

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("at least one prediction/reference pair is required")
    distances = []
    for prediction, reference in zip(predictions, references, strict=True):
        prediction_tokens = prediction.split()
        reference_tokens = reference.split()
        denominator = max(len(prediction_tokens), len(reference_tokens), 1)
        distances.append(_edit_distance(prediction_tokens, reference_tokens) / denominator)
    return sum(distances) / len(distances)


__all__ = ["corpus_bleu", "normalized_edit_distance"]
