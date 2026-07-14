"""Run the trained Smart-Log models against a newly uploaded HDFS log.

This is the "inference" counterpart to main.py's training run: it loads the
already-trained Isolation Forest, One-Class SVM, and StandardScaler from
models/, parses a user-supplied log file with the exact same streaming
parser and feature engineering used for training, scores every block, and
produces alerts/plots for that run only (kept isolated per job under
outputs/uploads/<job_id>/ so concurrent/successive uploads never clobber
each other or the original training run's outputs).

If the caller also supplies a ground-truth labels CSV (same two-column
"BlockId,Label" format as the training dataset's anomaly_label.csv), full
evaluation metrics and confusion matrices are computed too; otherwise the
run reports detections only, which is the expected case for a genuinely new,
unlabeled log.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

from src.alert_generator import generate_alerts, save_alerts
from src.evaluator import build_comparison_table, evaluate_strategy, select_best_strategy
from src.feature_engineering import ALL_EVENT_IDS, clean_feature_matrix, get_feature_columns, build_feature_matrix
from src.hybrid_detector import hybrid_and, hybrid_or
from src.isolation_forest_model import evaluate_isolation_forest
from src.one_class_svm_model import evaluate_one_class_svm
from src.shap_explainer import build_explanation_sentence, build_shap_explainer, compute_shap_values, top_contributing_features
from src import visualization as viz

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

ProgressFn = Callable[[str], None]


class ModelsNotTrainedError(RuntimeError):
    """Raised when models/*.joblib are missing (main.py has not been run yet)."""


@dataclass
class InferenceResult:
    job_id: str
    n_lines_parsed: int
    n_blocks: int
    has_labels: bool
    comparison_table: Optional[pd.DataFrame]
    best_strategy_name: Optional[str]
    alerts: list
    alerts_csv_path: Path
    alerts_json_path: Path
    plots_dir: Path
    plot_files: list[str]
    severities_count: dict[str, int]


def _load_models():
    paths = {
        "isolation_forest": MODELS_DIR / "isolation_forest.joblib",
        "one_class_svm": MODELS_DIR / "one_class_svm.joblib",
        "scaler": MODELS_DIR / "scaler.joblib",
    }
    missing = [name for name, p in paths.items() if not p.is_file()]
    if missing:
        raise ModelsNotTrainedError(
            "Trained models not found: "
            + ", ".join(missing)
            + ". Run `python main.py` once against the training dataset first."
        )
    return {name: joblib.load(p) for name, p in paths.items()}


def load_uploaded_labels(labels_path: Path) -> pd.DataFrame:
    """Parse an uploaded ground-truth labels CSV (BlockId,Label columns).

    Mirrors src.data_loader.load_anomaly_labels' normalization (lowercase
    column names, 0/1 is_anomaly flag) so merge_labels-style logic can reuse
    it, but is kept separate from the training data loader since this file
    comes from a user upload rather than the fixed dataset path.
    """
    df = pd.read_csv(labels_path)
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        if col.lower() in ("blockid", "block_id"):
            rename_map[col] = "block_id"
        elif col.lower() == "label":
            rename_map[col] = "label"
    df = df.rename(columns=rename_map)
    if "block_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Labels CSV must have 'BlockId' and 'Label' columns")
    df["label"] = df["label"].astype(str).str.strip()
    df["is_anomaly"] = (df["label"].str.lower() == "anomaly").astype(int)
    return df[["block_id", "label", "is_anomaly"]]


def run_inference(
    job_id: str,
    log_path: Path,
    output_root: Path,
    labels_path: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
) -> InferenceResult:
    """Score an uploaded HDFS log with the trained models and produce alerts/plots."""

    def report(msg: str) -> None:
        if progress:
            progress(msg)

    job_dir = output_root / job_id
    plots_dir = job_dir / "plots"
    alerts_dir = job_dir / "alerts"
    metrics_dir = job_dir / "metrics"
    for d in (plots_dir, alerts_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    report("Loading trained models...")
    models = _load_models()
    if_model = models["isolation_forest"]
    ocsvm_model = models["one_class_svm"]
    scaler = models["scaler"]

    report("Parsing uploaded log file...")
    features_df = build_feature_matrix(log_path, progress_every=500_000)
    n_blocks_parsed = len(features_df)
    if n_blocks_parsed == 0:
        raise ValueError(
            "No valid HDFS log lines were parsed from the uploaded file. "
            "Expected the '<date> <time> <pid> <level> <component>: <message>' "
            "format with a blk_* id in each message."
        )

    has_labels = labels_path is not None
    if has_labels:
        report("Merging uploaded ground-truth labels...")
        labels_df = load_uploaded_labels(labels_path)
        merged_df = features_df.merge(labels_df, on="block_id", how="inner")
        if merged_df.empty:
            raise ValueError("None of the uploaded log's block ids matched any row in the labels file.")
    else:
        merged_df = features_df

    report("Extracting behavioral features...")
    merged_df = clean_feature_matrix(merged_df)
    feature_cols = get_feature_columns(merged_df)

    # Feature columns must match the scaler's training-time column order.
    # build_feature_matrix always emits the same fixed column set/order
    # (base features, then event_freq_<E1..E29,E_UNKNOWN>), identical to
    # what prepare_train_test_split saw during training, so no extra
    # reordering step is needed beyond using get_feature_columns' output.
    X = scaler.transform(merged_df[feature_cols].values)
    block_ids = merged_df["block_id"].values

    report("Scoring with Isolation Forest...")
    if_result = evaluate_isolation_forest(if_model, X)

    report("Scoring with One-Class SVM...")
    ocsvm_result = evaluate_one_class_svm(ocsvm_model, X)

    and_predictions = hybrid_and(if_result.predictions, ocsvm_result.predictions)
    or_predictions = hybrid_or(if_result.predictions, ocsvm_result.predictions)

    comparison_table = None
    best_name = "Isolation Forest"
    results = None
    if has_labels:
        report("Evaluating against ground truth...")
        y_true = merged_df["is_anomaly"].values
        results = [
            evaluate_strategy("Isolation Forest", y_true, if_result.predictions),
            evaluate_strategy("One-Class SVM", y_true, ocsvm_result.predictions),
            evaluate_strategy("Hybrid AND", y_true, and_predictions),
            evaluate_strategy("Hybrid OR", y_true, or_predictions),
        ]
        comparison_table = build_comparison_table(results)
        best = select_best_strategy(results)
        best_name = best.name
        comparison_table.to_csv(metrics_dir / "model_comparison.csv", index=False)
        for r in results:
            np.savetxt(
                metrics_dir / f"confusion_matrix_{r.name.replace(' ', '_')}.csv",
                r.confusion, fmt="%d", delimiter=",",
            )

    report("Generating SHAP explanations for flagged anomalies...")
    explainer = build_shap_explainer(if_model)
    anomalous_idx = np.where(if_result.predictions == 1)[0]
    if len(anomalous_idx) > 0:
        shap_explanation = compute_shap_values(explainer, X[anomalous_idx], feature_cols)
        top_features_per_row = [
            top_contributing_features(shap_explanation.shap_values[i], feature_cols)
            for i in range(len(anomalous_idx))
        ]
        explanations = [
            build_explanation_sentence(block_ids[anomalous_idx[i]], top_features_per_row[i])
            for i in range(len(anomalous_idx))
        ]
    else:
        shap_explanation = None
        top_features_per_row = []
        explanations = []

    report("Generating security alerts...")
    alerts = generate_alerts(
        block_ids=block_ids[anomalous_idx],
        predictions=np.ones(len(anomalous_idx), dtype=int),
        anomaly_scores=if_result.anomaly_scores[anomalous_idx],
        top_features_per_row=top_features_per_row,
        explanations=explanations,
    )
    csv_path, json_path = save_alerts(alerts, alerts_dir)

    report("Rendering visualizations...")
    plot_files: list[str] = []
    if has_labels:
        viz.plot_label_distribution(merged_df, plots_dir)
        plot_files.append("01_label_distribution.png")
    viz.plot_event_frequency_distribution(merged_df, plots_dir)
    plot_files.append("02_event_frequency_distribution.png")
    if has_labels:
        viz.plot_sequence_length_distribution(merged_df, plots_dir)
        plot_files.append("03_sequence_length_distribution.png")
        viz.plot_anomaly_score_distribution(if_result.anomaly_scores, merged_df["is_anomaly"].values, plots_dir)
        plot_files.append("04_isolation_forest_score_distribution.png")
        for r in results:
            fname = f"05_confusion_matrix_{r.name.replace(' ', '_')}.png"
            viz.plot_confusion_matrix(r.confusion, f"Confusion Matrix - {r.name}", plots_dir, fname)
            plot_files.append(fname)
        viz.plot_model_comparison(comparison_table, plots_dir)
        plot_files.append("06_model_comparison.png")
    if shap_explanation is not None and len(anomalous_idx) > 1:
        viz.plot_shap_summary(shap_explanation.shap_values, X[anomalous_idx], feature_cols, plots_dir)
        plot_files.append("07_shap_summary.png")
    severities_count: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    if alerts:
        severities = [a.severity for a in alerts]
        for s in severities:
            severities_count[s] = severities_count.get(s, 0) + 1
        viz.plot_severity_distribution(severities, plots_dir)
        plot_files.append("08_severity_distribution.png")
        top_features_flat = [f for row in top_features_per_row for f in row]
        if top_features_flat:
            viz.plot_top_contributing_features(top_features_flat, plots_dir)
            plot_files.append("09_top_contributing_features.png")

    report("Done.")
    return InferenceResult(
        job_id=job_id,
        n_lines_parsed=int(merged_df["total_events"].sum()),
        n_blocks=n_blocks_parsed,
        has_labels=has_labels,
        comparison_table=comparison_table,
        best_strategy_name=best_name,
        alerts=alerts,
        alerts_csv_path=csv_path,
        alerts_json_path=json_path,
        plots_dir=plots_dir,
        plot_files=plot_files,
        severities_count=severities_count,
    )
