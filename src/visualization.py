"""Generate matplotlib/seaborn visualizations for the Smart-Log pipeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _savefig(fig, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_label_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Normal vs anomaly distribution."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="label", data=df, ax=ax, hue="label", palette=["#4C72B0", "#C44E52"], legend=False)
    ax.set_title("Normal vs Anomaly Block Distribution")
    ax.set_xlabel("")
    _savefig(fig, output_dir, "01_label_distribution.png")


def plot_event_frequency_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Event frequency distribution (total_events per block)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["total_events"], bins=50, ax=ax, color="#4C72B0")
    ax.set_title("Event Frequency Distribution (events per block)")
    ax.set_xlabel("Total events per block")
    _savefig(fig, output_dir, "02_event_frequency_distribution.png")


def plot_sequence_length_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Sequence length distribution split by label."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="total_events", hue="label", bins=50, ax=ax, log_scale=(False, True))
    ax.set_title("Sequence Length Distribution by Label")
    ax.set_xlabel("Total events per block")
    _savefig(fig, output_dir, "03_sequence_length_distribution.png")


def plot_anomaly_score_distribution(scores: np.ndarray, y_true: np.ndarray, output_dir: Path) -> None:
    """Isolation Forest anomaly-score distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    df = pd.DataFrame({"score": scores, "label": np.where(y_true == 1, "Anomaly", "Normal")})
    sns.histplot(data=df, x="score", hue="label", bins=50, ax=ax)
    ax.set_title("Isolation Forest Anomaly Score Distribution")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    _savefig(fig, output_dir, "04_isolation_forest_score_distribution.png")


def plot_confusion_matrix(confusion: np.ndarray, title: str, output_dir: Path, filename: str) -> None:
    """Confusion matrix heatmap for a single strategy."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _savefig(fig, output_dir, filename)


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart comparing Accuracy/Precision/Recall/F1 across strategies."""
    melted = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model / Strategy Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    _savefig(fig, output_dir, "06_model_comparison.png")


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names: list[str], output_dir: Path) -> None:
    """SHAP summary plot for the Isolation Forest explainer."""
    import shap

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=15)
    fig = plt.gcf()
    _savefig(fig, output_dir, "07_shap_summary.png")


def plot_severity_distribution(severities: list[str], output_dir: Path) -> None:
    """Severity distribution of generated alerts."""
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=severities, order=order, ax=ax, hue=severities, palette="rocket", legend=False)
    ax.set_title("Alert Severity Distribution")
    _savefig(fig, output_dir, "08_severity_distribution.png")


def plot_top_contributing_features(top_features_flat: list[str], output_dir: Path, top_n: int = 15) -> None:
    """Bar chart of the most frequently top-contributing features across alerts."""
    counts = pd.Series(top_features_flat).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, hue=counts.index, palette="mako", legend=False)
    ax.set_title("Top Anomaly-Contributing Features (across alerts)")
    ax.set_xlabel("Times appearing in an alert's top features")
    _savefig(fig, output_dir, "09_top_contributing_features.png")
