"""Smart-Log: Automated Syslog Anomaly Detection for Server Security.

Runs the full pipeline: dataset inspection -> log parsing -> preprocessing
-> feature engineering -> Isolation Forest -> One-Class SVM -> hybrid
strategy evaluation -> SHAP explanations -> alert generation -> evaluation
and visualizations.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.alert_generator import generate_alerts, save_alerts
from src.data_loader import (
    HDFS_LOG_PATH,
    check_dataset_files,
    load_anomaly_labels,
)
from src.evaluator import build_comparison_table, evaluate_strategy, select_best_strategy
from src.feature_engineering import build_feature_matrix
from src.hybrid_detector import hybrid_and, hybrid_or
from src.isolation_forest_model import evaluate_isolation_forest, train_isolation_forest
from src.one_class_svm_model import evaluate_one_class_svm, train_one_class_svm
from src.preprocessing import merge_labels, prepare_train_test_split, run_data_quality_checks
from src.shap_explainer import (
    build_explanation_sentence,
    build_shap_explainer,
    compute_shap_values,
    top_contributing_features,
)
from src import visualization as viz

PROJECT_ROOT = Path(__file__).parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
ALERTS_DIR = OUTPUTS_DIR / "alerts"
METRICS_DIR = OUTPUTS_DIR / "metrics"

RANDOM_STATE = 42


def main() -> None:
    missing = check_dataset_files()
    if missing:
        print("SMART-LOG: Required dataset files are missing from data/raw/:")
        for path in missing:
            print(f"  - {path.name}")
        print("\nPlace the HDFS Log Anomaly Dataset files there and re-run this script.")
        return

    print("[1/10] Loading HDFS dataset...")
    labels_df = load_anomaly_labels()
    print(f"  Loaded {len(labels_df):,} ground-truth block labels")

    print("[2/10] Parsing logs...")
    raw_features_cache_path = DATA_PROCESSED_DIR / "block_features_raw.parquet"
    if raw_features_cache_path.is_file():
        print(f"  Found cached parsed features at {raw_features_cache_path}, skipping raw log re-parse")
        features_df = pd.read_parquet(raw_features_cache_path)
    else:
        features_df = build_feature_matrix(HDFS_LOG_PATH)
        features_df.to_parquet(raw_features_cache_path, index=False)

    print("[3/10] Preprocessing records...")
    merged_df = merge_labels(features_df, labels_df)
    run_data_quality_checks(merged_df)
    merged_df.to_parquet(DATA_PROCESSED_DIR / "block_features.parquet", index=False)
    print(f"  Saved processed feature matrix to {DATA_PROCESSED_DIR / 'block_features.parquet'}")

    print("[4/10] Extracting behavioral features...")
    print(f"  Feature matrix shape: {merged_df.shape[0]:,} blocks x "
          f"{merged_df.shape[1] - 3} behavioral features")
    prepared = prepare_train_test_split(merged_df, random_state=RANDOM_STATE)
    print(f"  Train set (normal-only): {prepared.X_train.shape[0]:,} blocks")
    print(f"  Test set (mixed): {prepared.X_test.shape[0]:,} blocks "
          f"({int(prepared.y_test.sum()):,} anomalies)")

    print("[5/10] Training Isolation Forest...")
    if_model = train_isolation_forest(prepared.X_train, random_state=RANDOM_STATE)
    if_result = evaluate_isolation_forest(if_model, prepared.X_test)
    joblib.dump(if_model, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(prepared.scaler, MODELS_DIR / "scaler.joblib")

    print("[6/10] Training One-Class SVM...")
    ocsvm_model = train_one_class_svm(prepared.X_train)
    ocsvm_result = evaluate_one_class_svm(ocsvm_model, prepared.X_test)
    joblib.dump(ocsvm_model, MODELS_DIR / "one_class_svm.joblib")

    print("[7/10] Evaluating hybrid strategies...")
    and_predictions = hybrid_and(if_result.predictions, ocsvm_result.predictions)
    or_predictions = hybrid_or(if_result.predictions, ocsvm_result.predictions)

    results = [
        evaluate_strategy("Isolation Forest", prepared.y_test, if_result.predictions),
        evaluate_strategy("One-Class SVM", prepared.y_test, ocsvm_result.predictions),
        evaluate_strategy("Hybrid AND", prepared.y_test, and_predictions),
        evaluate_strategy("Hybrid OR", prepared.y_test, or_predictions),
    ]
    comparison_table = build_comparison_table(results)
    best = select_best_strategy(results)
    print(comparison_table.to_string(index=False))
    print(f"  Best strategy by F1-Score: {best.name}")

    print("[8/10] Generating SHAP explanations...")
    explainer = build_shap_explainer(if_model)
    anomalous_idx = np.where(if_result.predictions == 1)[0]
    shap_sample_idx = anomalous_idx  # explain every IF-flagged row
    shap_explanation = compute_shap_values(explainer, prepared.X_test[shap_sample_idx], prepared.feature_columns)

    top_features_per_row = [
        top_contributing_features(shap_explanation.shap_values[i], prepared.feature_columns)
        for i in range(len(shap_sample_idx))
    ]
    explanations = [
        build_explanation_sentence(prepared.test_block_ids[shap_sample_idx[i]], top_features_per_row[i])
        for i in range(len(shap_sample_idx))
    ]

    print("[9/10] Generating security alerts...")
    alerts = generate_alerts(
        block_ids=prepared.test_block_ids[shap_sample_idx],
        predictions=np.ones(len(shap_sample_idx), dtype=int),
        anomaly_scores=if_result.anomaly_scores[shap_sample_idx],
        top_features_per_row=top_features_per_row,
        explanations=explanations,
    )
    csv_path, json_path = save_alerts(alerts, ALERTS_DIR)
    print(f"  Generated {len(alerts):,} alerts -> {csv_path.name}, {json_path.name}")

    print("[10/10] Saving evaluation results...")
    comparison_table.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    for r in results:
        np.savetxt(METRICS_DIR / f"confusion_matrix_{r.name.replace(' ', '_')}.csv", r.confusion, fmt="%d", delimiter=",")

    print("  Generating visualizations...")
    viz.plot_label_distribution(merged_df, PLOTS_DIR)
    viz.plot_event_frequency_distribution(merged_df, PLOTS_DIR)
    viz.plot_sequence_length_distribution(merged_df, PLOTS_DIR)
    viz.plot_anomaly_score_distribution(if_result.anomaly_scores, prepared.y_test, PLOTS_DIR)
    for r in results:
        viz.plot_confusion_matrix(
            r.confusion, f"Confusion Matrix - {r.name}", PLOTS_DIR,
            f"05_confusion_matrix_{r.name.replace(' ', '_')}.png",
        )
    viz.plot_model_comparison(comparison_table, PLOTS_DIR)
    viz.plot_shap_summary(
        shap_explanation.shap_values, prepared.X_test[shap_sample_idx], prepared.feature_columns, PLOTS_DIR
    )
    severities = [a.severity for a in alerts]
    viz.plot_severity_distribution(severities, PLOTS_DIR)
    top_features_flat = [f for row in top_features_per_row for f in row]
    viz.plot_top_contributing_features(top_features_flat, PLOTS_DIR)

    print("\nSMART-LOG EXECUTION COMPLETED\n")
    print(f"Total log sequences (blocks): {len(merged_df):,}")
    print(f"Normal sequences: {int((merged_df['is_anomaly'] == 0).sum()):,}")
    print(f"Actual anomalies: {int((merged_df['is_anomaly'] == 1).sum()):,}")
    print(f"Predicted anomalies ({best.name}): {int(best.predictions.sum()):,}")
    print(f"Best model/strategy: {best.name}")
    print(f"Accuracy: {best.accuracy:.4f}")
    print(f"Precision: {best.precision:.4f}")
    print(f"Recall: {best.recall:.4f}")
    print(f"F1-Score: {best.f1:.4f}")
    print(f"Number of generated alerts: {len(alerts):,}")


if __name__ == "__main__":
    main()
