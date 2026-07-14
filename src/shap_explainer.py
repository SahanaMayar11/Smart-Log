"""SHAP explainability for the primary anomaly detector (Isolation Forest).

Isolation Forest is the project's primary anomaly detector, so SHAP is used
to explain *its* decisions specifically (not the One-Class SVM, which has no
tree structure SHAP's fast explainers can use).

Compatibility: scikit-learn's IsolationForest is a tree-ensemble model, so
`shap.TreeExplainer` supports it directly (verified interactively before
writing this module: `shap.TreeExplainer(IsolationForest(...).fit(X))`
succeeds and returns per-feature SHAP values with the same shape as the
input). This avoids misusing SHAP on a model type it does not support.

What the SHAP values represent here: TreeExplainer explains the model's
raw anomaly path-length score (the same quantity underlying
`decision_function`/`score_samples`). For each row, a feature's SHAP value
is its signed contribution to that row's score relative to the model's
average score across the background/training data: a large negative SHAP
value pushes the score toward "more anomalous" (Isolation Forest scores
anomalies lower), so features with the most negative SHAP values are the
top contributors to a row being flagged as an anomaly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest


@dataclass
class ShapExplanation:
    shap_values: np.ndarray  # shape (n_rows, n_features)
    feature_names: list[str]


def build_shap_explainer(model: IsolationForest) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for a fitted Isolation Forest."""
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X: np.ndarray, feature_names: list[str]) -> ShapExplanation:
    """Compute SHAP values for the given rows."""
    shap_values = explainer.shap_values(X)
    return ShapExplanation(shap_values=np.asarray(shap_values), feature_names=feature_names)


def top_contributing_features(shap_row: np.ndarray, feature_names: list[str], top_n: int = 3) -> list[str]:
    """Return the top-N feature names most responsible for an anomalous score.

    Ranks by the most negative SHAP value (Isolation Forest: lower score =
    more anomalous, so the most negative contributions drove the anomaly).
    """
    order = np.argsort(shap_row)  # ascending: most negative first
    return [feature_names[i] for i in order[:top_n]]


_FEATURE_PHRASES = {
    "total_events": "unusually high event frequency",
    "unique_events": "an abnormal number of distinct event types",
    "event_diversity": "abnormal event sequence diversity",
    "error_event_count": "a high error event count",
    "warn_event_count": "a high warning event count",
    "info_event_count": "an abnormal info event count",
    "event_transitions": "abnormal event transition patterns",
    "repeated_event_count": "an abnormal number of repeated events",
    "time_span_seconds": "abnormal sequence duration",
    "avg_inter_event_seconds": "abnormal timing between events",
    "distinct_components": "an abnormal number of components involved",
}


def humanize_feature_name(name: str) -> str:
    """Turn a feature column name into a short human-readable phrase."""
    if name in _FEATURE_PHRASES:
        return _FEATURE_PHRASES[name]
    if name.startswith("event_freq_"):
        event_id = name.replace("event_freq_", "")
        return f"an abnormal frequency of {event_id}-type events"
    return name.replace("_", " ")


def build_explanation_sentence(block_id: str, top_features: list[str]) -> str:
    """Build a human-readable explanation sentence for one flagged block."""
    readable = [humanize_feature_name(f) for f in top_features]
    if not readable:
        return f"Block {block_id} was flagged as anomalous, but no dominant contributing feature was identified."
    if len(readable) == 1:
        factors = readable[0]
    elif len(readable) == 2:
        factors = f"{readable[0]} and {readable[1]}"
    else:
        factors = ", ".join(readable[:-1]) + f", and {readable[-1]}"
    return (
        f"This log sequence was classified as anomalous primarily because of "
        f"{factors} compared with normal HDFS block behavior."
    )
