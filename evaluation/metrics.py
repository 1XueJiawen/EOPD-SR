"""
Evaluation Metrics Module (§4.2)
=================================
Implements all metrics used in the paper:
  - F1 Score: Token-level F1 between predicted and gold answers
  - Hits@1: Whether the top-1 prediction matches any gold answer
  - ReCall@50: Recall of gold answers within top-50 retrieved subgraph entities
  - Avg. Graph Size: Average number of triples in retrieved subgraphs
  - Percentage Reduction: Compared to Graph-only baseline

Table 1: EPOSR achieves superior performance across 4 benchmark datasets.
"""

import re
import string
import json
from collections import Counter
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    sample_id: str
    question: str
    predicted: str
    predicted_entities: list[str]
    gold_answers: list[str]
    f1: float = 0.0
    hits_at_1: float = 0.0
    graph_size: int = 0


@dataclass
class AggregateMetrics:
    """Aggregated evaluation metrics."""
    f1: float = 0.0
    hits_at_1: float = 0.0
    recall_at_50: float = 0.0
    avg_graph_size: float = 0.0
    pct_reduction: float = 0.0
    total_samples: int = 0


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    Lower case, remove articles, punctuation, and extra whitespace.
    Based on SQuAD evaluation script.
    """
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()


def token_f1(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 between prediction and gold answer.
    Standard metric for KBQA evaluation.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_f1(prediction: str, gold_answers: list[str]) -> float:
    """
    Compute F1 score against all gold answers, return the maximum.

    Args:
        prediction: Predicted answer text.
        gold_answers: List of gold answer texts.

    Returns:
        Maximum F1 score across all gold answers.
    """
    if not gold_answers:
        return 0.0

    return max(token_f1(prediction, gold) for gold in gold_answers)


def compute_hits_at_1(prediction: str, gold_answers: list[str]) -> float:
    """
    Compute Hits@1: 1.0 if the top-1 prediction matches any gold answer.

    Exact match after normalization.
    """
    pred_norm = normalize_answer(prediction)
    for gold in gold_answers:
        if normalize_answer(gold) in pred_norm or pred_norm in normalize_answer(gold):
            return 1.0
    return 0.0


def compute_recall_at_k(
    predicted_entities: list[str],
    gold_answers: list[str],
    k: int = 50,
) -> float:
    """
    Compute Recall@k: fraction of gold answers found in top-k predictions.

    Args:
        predicted_entities: List of predicted entity names.
        gold_answers: List of gold answer entity names.
        k: Number of top predictions to consider.

    Returns:
        Recall score.
    """
    if not gold_answers:
        return 0.0

    pred_set = {normalize_answer(e) for e in predicted_entities[:k]}
    gold_set = {normalize_answer(a) for a in gold_answers}

    hits = sum(1 for g in gold_set if any(g in p or p in g for p in pred_set))
    return hits / len(gold_set)


def evaluate_sample(
    sample_id: str,
    question: str,
    predicted: str,
    predicted_entities: list[str],
    gold_answers: list[str],
    graph_size: int = 0,
) -> EvalResult:
    """
    Evaluate a single QA sample.

    Returns:
        EvalResult with all metrics computed.
    """
    return EvalResult(
        sample_id=sample_id,
        question=question,
        predicted=predicted,
        predicted_entities=predicted_entities,
        gold_answers=gold_answers,
        f1=compute_f1(predicted, gold_answers),
        hits_at_1=compute_hits_at_1(predicted, gold_answers),
        graph_size=graph_size,
    )


def evaluate_predictions(
    predictions: list[dict],
    gold_answers: list[dict],
) -> AggregateMetrics:
    """
    Evaluate a batch of predictions against gold answers.

    Args:
        predictions: List of dicts with keys: id, answer, answer_entities, graph_size
        gold_answers: List of dicts with keys: id, answer (list of strings)

    Returns:
        AggregateMetrics with averaged scores.
    """
    # Match predictions with gold answers by ID
    gold_map = {g["id"]: g for g in gold_answers}

    f1_scores = []
    hits_scores = []
    recall_scores = []
    graph_sizes = []

    for pred in predictions:
        sample_id = pred["id"]
        if sample_id not in gold_map:
            continue

        gold = gold_map[sample_id]
        gold_ans = gold["answer"]
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]

        result = evaluate_sample(
            sample_id=sample_id,
            question=pred.get("question", ""),
            predicted=pred["answer"],
            predicted_entities=pred.get("answer_entities", []),
            gold_answers=gold_ans,
            graph_size=pred.get("graph_size", 0),
        )

        f1_scores.append(result.f1)
        hits_scores.append(result.hits_at_1)
        recall_scores.append(
            compute_recall_at_50(pred.get("answer_entities", []), gold_ans)
        )
        graph_sizes.append(result.graph_size)

    n = len(f1_scores)
    if n == 0:
        return AggregateMetrics()

    return AggregateMetrics(
        f1=np.mean(f1_scores),
        hits_at_1=np.mean(hits_scores),
        recall_at_50=np.mean(recall_scores),
        avg_graph_size=np.mean(graph_sizes),
        total_samples=n,
    )


def compute_recall_at_50(predicted_entities: list[str], gold_answers: list[str]) -> float:
    """Compute Recall@50."""
    return compute_recall_at_k(predicted_entities, gold_answers, k=50)


def compute_metrics(
    predictions: list[dict],
    gold_answers: list[dict],
    baseline_graph_size: Optional[float] = None,
) -> dict:
    """
    Compute all metrics and return as a dictionary.

    Args:
        predictions: List of prediction dicts.
        gold_answers: List of gold answer dicts.
        baseline_graph_size: Graph-only baseline size for computing reduction.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = evaluate_predictions(predictions, gold_answers)

    result = {
        "F1": round(metrics.f1 * 100, 2),
        "Hits@1": round(metrics.hits_at_1 * 100, 2),
        "ReCall@50": round(metrics.recall_at_50 * 100, 2),
        "Avg. Graph Size": round(metrics.avg_graph_size, 1),
        "Total Samples": metrics.total_samples,
    }

    if baseline_graph_size and baseline_graph_size > 0:
        pct_reduction = (1 - metrics.avg_graph_size / baseline_graph_size) * 100
        result["Pct. Reduction"] = round(pct_reduction, 1)

    return result


def save_results(results: dict, output_path: str):
    """Save evaluation results to JSON file."""
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")
