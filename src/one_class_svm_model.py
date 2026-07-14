"""One-Class SVM: secondary anomaly validation / refinement model."""

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM


@dataclass
class OneClassSVMResult:
    model: OneClassSVM
    predictions: np.ndarray  # 0 = normal, 1 = anomaly
    anomaly_scores: np.ndarray  # higher = more anomalous
    decision_scores: np.ndarray  # raw sklearn decision_function output


def subsample_for_training(X_train: np.ndarray, max_samples: int, random_state: int = 42) -> np.ndarray:
    """Randomly subsample the training set for One-Class SVM fitting.

    RBF-kernel SVMs have O(n^2)-O(n^3) training complexity, which makes
    fitting on the full ~450,000-row normal training set computationally
    infeasible. A fixed-size random subsample (default 10,000 rows, capped
    by `max_samples`) is used instead so training completes in a reasonable
    time, while Isolation Forest (which scales far better) is still trained
    on the full training set. This is a documented engineering tradeoff, not
    a change to the model or its evaluation: predictions/evaluation still
    run on the full held-out test set.
    """
    if X_train.shape[0] <= max_samples:
        return X_train
    rng = np.random.RandomState(random_state)
    idx = rng.choice(X_train.shape[0], size=max_samples, replace=False)
    return X_train[idx]


def train_one_class_svm(
    X_train: np.ndarray,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    nu: float = 0.05,
    max_train_samples: int = 10_000,
    random_state: int = 42,
) -> OneClassSVM:
    """Fit a One-Class SVM on the (assumed-normal, StandardScaler-scaled) training features.

    `nu` upper-bounds the fraction of training points the model is allowed
    to treat as outliers/support vectors; kept small since the training
    split is predominantly normal data. See `subsample_for_training` for why
    training uses a bounded random subsample rather than the full split.
    """
    X_fit = subsample_for_training(X_train, max_train_samples, random_state=random_state)
    model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    model.fit(X_fit)
    return model


def evaluate_one_class_svm(model: OneClassSVM, X: np.ndarray) -> OneClassSVMResult:
    """Score data with a fitted One-Class SVM and produce 0/1 predictions.

    scikit-learn's OneClassSVM.predict returns 1 for normal and -1 for
    anomalous; converted here to the project's 0 = Normal, 1 = Anomaly
    convention.
    """
    raw_predictions = model.predict(X)  # 1 = normal, -1 = anomaly
    predictions = np.where(raw_predictions == -1, 1, 0)

    decision_scores = model.decision_function(X)  # higher = more normal
    anomaly_scores = -decision_scores

    return OneClassSVMResult(
        model=model,
        predictions=predictions,
        anomaly_scores=anomaly_scores,
        decision_scores=decision_scores,
    )
