"""Data preparation: merge labels, validate, split, and scale features.

Handles missing-value analysis, duplicate analysis, invalid record handling,
feature validation, numerical feature selection, scaling, and the
train/test split strategy, all on the per-block feature matrix produced by
feature_engineering.build_feature_matrix.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import clean_feature_matrix, get_feature_columns


def merge_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach ground-truth labels to the feature matrix by block_id.

    Uses an inner join: only blocks present in both the parsed log data and
    the ground-truth label file are kept, since a block missing from either
    side cannot be evaluated or is not part of the labeled population.
    """
    merged = features.merge(labels[["block_id", "label", "is_anomaly"]], on="block_id", how="inner")
    dropped = len(features) - len(merged)
    if dropped:
        print(f"  {dropped} blocks in the parsed logs have no ground-truth label and were dropped")
    return merged


def run_data_quality_checks(df: pd.DataFrame) -> None:
    """Print missing-value, duplicate, and invalid-record diagnostics."""
    feature_cols = get_feature_columns(df)
    print("Missing value analysis:")
    na_counts = df[feature_cols + ["is_anomaly"]].isna().sum()
    na_counts = na_counts[na_counts > 0]
    print(na_counts.to_string() if len(na_counts) else "  No missing values")

    print("Duplicate analysis:")
    dup_blocks = df["block_id"].duplicated().sum()
    dup_rows = df.duplicated(subset=feature_cols).sum()
    print(f"  Duplicate block_id rows: {dup_blocks}")
    print(f"  Rows with fully duplicate feature values: {dup_rows}")

    print("Invalid record check (negative counts/durations):")
    invalid_mask = (df[feature_cols].select_dtypes(include=[np.number]) < 0).any(axis=1)
    print(f"  Rows with a negative feature value: {int(invalid_mask.sum())}")


@dataclass
class PreparedData:
    """Holds the scaled train/test split used by the anomaly models."""

    feature_columns: list[str]
    scaler: StandardScaler
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_block_ids: np.ndarray
    train_block_ids: np.ndarray


def prepare_train_test_split(
    df: pd.DataFrame,
    normal_train_fraction: float = 0.8,
    random_state: int = 42,
) -> PreparedData:
    """Split and scale the feature matrix for semi-supervised anomaly detection.

    Training strategy (documented explicitly, since this is an unsupervised
    anomaly detection project):

    Isolation Forest and One-Class SVM are novelty-detection style models:
    they are designed to be fit on data that represents "normal" behavior,
    then flag deviations from it. Ground-truth labels are used here only to
    select *which rows* go into the training split (a data-split decision),
    never as a model input feature/column, so the models themselves never
    see the anomaly labels during fitting - satisfying "do not train the
    unsupervised models directly using anomaly labels".

    Concretely:
    - `normal_train_fraction` (default 80%) of the Normal-labeled blocks
      form the training set.
    - The remaining Normal blocks plus every Anomaly-labeled block form the
      held-out test set used for evaluation.

    This avoids data leakage (no anomaly-labeled block is ever used for
    fitting) and mirrors how these models would be used in production: fit
    on a known-good historical window, then score new activity.
    """
    feature_cols = get_feature_columns(df)
    df = clean_feature_matrix(df)

    normal_df = df[df["is_anomaly"] == 0]
    anomaly_df = df[df["is_anomaly"] == 1]

    normal_train, normal_test = train_test_split(
        normal_df, train_size=normal_train_fraction, random_state=random_state, shuffle=True
    )

    test_df = pd.concat([normal_test, anomaly_df], axis=0).sample(frac=1.0, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_train[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    return PreparedData(
        feature_columns=feature_cols,
        scaler=scaler,
        X_train=X_train,
        X_test=X_test,
        y_test=test_df["is_anomaly"].values,
        test_block_ids=test_df["block_id"].values,
        train_block_ids=normal_train["block_id"].values,
    )
