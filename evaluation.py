import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def classification_metrics(tp, fp, fn, tn):
    """Calculate class-imbalance-aware metrics from confusion counts."""
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        'accuracy': (tp + tn) / total if total else 0.0,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'balanced_accuracy': (recall + specificity) / 2,
        'coverage': (tp + fp) / total if total else 0.0,
    }


def probability_metrics(labels, scores):
    """Calculate ranking and calibration metrics for probability scores."""
    return {
        'average_precision': average_precision_score(labels, scores),
        'brier_score': brier_score_loss(labels, scores),
    }


def backtest_top_scores(scores, realized_returns, fraction=0.1, transaction_cost=0.0):
    """Compare an equal-weight top-score portfolio with the full universe."""
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be between 0 and 1')
    scores = np.asarray(scores)
    realized_returns = np.asarray(realized_returns)
    if scores.shape != realized_returns.shape or scores.size == 0:
        raise ValueError('scores and realized_returns must have equal, non-zero shapes')

    selection_count = max(1, int(np.ceil(scores.size * fraction)))
    selected = np.argpartition(scores, -selection_count)[-selection_count:]
    gross_return = float(realized_returns[selected].mean())
    benchmark_return = float(realized_returns.mean())
    return {
        'selected_count': selection_count,
        'gross_return': gross_return,
        'net_return': gross_return - transaction_cost,
        'benchmark_return': benchmark_return,
        'excess_return': gross_return - transaction_cost - benchmark_return,
    }
