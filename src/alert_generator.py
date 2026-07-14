"""Generate anomaly alerts with severity, top contributing features, and explanations.

Severity levels (LOW/MEDIUM/HIGH/CRITICAL) are a project-defined scheme, not
a claimed cybersecurity industry standard. Severity is derived from each
anomaly's normalized anomaly score (min-max scaled across all flagged
anomalies in the run) using fixed quartile-style thresholds, documented
below.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Project-defined severity thresholds on the min-max normalized anomaly
# score (0 = least anomalous of the flagged set, 1 = most anomalous).
SEVERITY_THRESHOLDS = [
    (0.25, "LOW"),
    (0.50, "MEDIUM"),
    (0.75, "HIGH"),
    (1.01, "CRITICAL"),
]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize anomaly scores to [0, 1]. Constant input maps to 0.5."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)


def score_to_severity(normalized_score: float) -> str:
    """Map a normalized anomaly score in [0, 1] to a project-defined severity level."""
    for threshold, label in SEVERITY_THRESHOLDS:
        if normalized_score <= threshold:
            return label
    return "CRITICAL"


@dataclass
class Alert:
    alert_id: str
    block_id: str
    anomaly: bool
    anomaly_score: float
    severity: str
    top_features: list[str]
    explanation: str


def generate_alerts(
    block_ids: np.ndarray,
    predictions: np.ndarray,
    anomaly_scores: np.ndarray,
    top_features_per_row: list[list[str]],
    explanations: list[str],
) -> list[Alert]:
    """Generate one Alert per row predicted as anomalous."""
    anomalous_idx = np.where(predictions == 1)[0]
    if len(anomalous_idx) == 0:
        return []

    normalized = normalize_scores(anomaly_scores[anomalous_idx])

    alerts = []
    for order, idx in enumerate(anomalous_idx):
        alerts.append(
            Alert(
                alert_id=f"ALT-{order + 1:04d}",
                block_id=str(block_ids[idx]),
                anomaly=True,
                anomaly_score=round(float(anomaly_scores[idx]), 6),
                severity=score_to_severity(normalized[order]),
                top_features=top_features_per_row[order],
                explanation=explanations[order],
            )
        )
    return alerts


def save_alerts(alerts: list[Alert], output_dir: Path) -> tuple[Path, Path]:
    """Save alerts to both CSV and JSON in output_dir. Returns (csv_path, json_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "alerts.csv"
    json_path = output_dir / "alerts.json"

    records = [asdict(a) for a in alerts]

    df = pd.DataFrame(records)
    if not df.empty:
        df["top_features"] = df["top_features"].apply(lambda x: ";".join(x))
    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    return csv_path, json_path
