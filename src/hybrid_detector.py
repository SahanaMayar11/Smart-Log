"""Hybrid validation logic combining Isolation Forest and One-Class SVM.

Implements and experimentally compares two hybrid strategies rather than
assuming either one is better:

- AND: anomaly only when both Isolation Forest and One-Class SVM agree.
- OR: anomaly when either model flags it.
"""

import numpy as np


def hybrid_and(if_predictions: np.ndarray, ocsvm_predictions: np.ndarray) -> np.ndarray:
    """Anomaly only if BOTH Isolation Forest and One-Class SVM flag it."""
    return ((if_predictions == 1) & (ocsvm_predictions == 1)).astype(int)


def hybrid_or(if_predictions: np.ndarray, ocsvm_predictions: np.ndarray) -> np.ndarray:
    """Anomaly if EITHER Isolation Forest or One-Class SVM flags it."""
    return ((if_predictions == 1) | (ocsvm_predictions == 1)).astype(int)
