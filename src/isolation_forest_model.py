"""Isolation Forest: primary anomaly detection model."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestResult:
    model: IsolationForest
    predictions: np.ndarray  # 0 = normal, 1 = anomaly
    anomaly_scores: np.ndarray  # higher = more anomalous
    decision_scores: np.ndarray  # raw sklearn decision_function output


def train_isolation_forest(
    X_train: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.03,
    max_samples: str | float = "auto",
    random_state: int = 42,
) -> IsolationForest:
    """Fit Isolation Forest on the (assumed-normal) training features.

    `contamination` is set to a small non-zero value (rather than 0) because
    the training split is *predominantly* normal but not guaranteed
    perfectly clean, and IsolationForest uses `contamination` to calibrate
    its internal decision threshold.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def evaluate_isolation_forest(model: IsolationForest, X: np.ndarray) -> IsolationForestResult:
    """Score data with a fitted Isolation Forest and produce 0/1 predictions.

    scikit-learn's IsolationForest.predict returns 1 for normal and -1 for
    anomalous; this is converted here to the project's 0 = Normal,
    1 = Anomaly convention.
    """
    raw_predictions = model.predict(X)  # 1 = normal, -1 = anomaly
    predictions = np.where(raw_predictions == -1, 1, 0)

    decision_scores = model.decision_function(X)  # higher = more normal
    anomaly_scores = -decision_scores  # higher = more anomalous

    return IsolationForestResult(
        model=model,
        predictions=predictions,
        anomaly_scores=anomaly_scores,
        decision_scores=decision_scores,
    )
