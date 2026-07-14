"""Evaluate anomaly detection strategies against ground-truth labels."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class StrategyMetrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    predictions: np.ndarray = field(repr=False)


def evaluate_strategy(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> StrategyMetrics:
    """Compute Accuracy, Precision, Recall, F1, and Confusion Matrix for one strategy."""
    return StrategyMetrics(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        predictions=y_pred,
    )


def build_comparison_table(results: list[StrategyMetrics]) -> pd.DataFrame:
    """Build the Model/Accuracy/Precision/Recall/F1 comparison table."""
    return pd.DataFrame(
        [
            {
                "Model": r.name,
                "Accuracy": r.accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1": r.f1,
            }
            for r in results
        ]
    )


def select_best_strategy(results: list[StrategyMetrics]) -> StrategyMetrics:
    """Select the best-performing strategy by F1-Score (balances precision and recall)."""
    return max(results, key=lambda r: r.f1)
