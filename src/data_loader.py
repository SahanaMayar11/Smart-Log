"""Load and inspect the raw HDFS Log Anomaly Dataset files.

This module is responsible only for locating and loading the raw dataset
files. It does not parse log lines (see log_parser.py) or engineer features
(see feature_engineering.py).
"""

from pathlib import Path

import pandas as pd

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
HDFS_LOG_PATH = DATA_RAW_DIR / "HDFS.log"
ANOMALY_LABEL_PATH = DATA_RAW_DIR / "anomaly_label.csv"
EVENT_TEMPLATES_PATH = DATA_RAW_DIR / "HDFS.log_templates.csv"

REQUIRED_FILES = [HDFS_LOG_PATH, ANOMALY_LABEL_PATH]


def check_dataset_files() -> list[Path]:
    """Return the list of required dataset files that are missing."""
    return [path for path in REQUIRED_FILES if not path.is_file()]


def load_anomaly_labels() -> pd.DataFrame:
    """Load the ground-truth per-block anomaly labels.

    Returns a DataFrame with columns:
        - block_id (str): HDFS Block ID, e.g. "blk_-1608999687919862906"
        - label (str): "Normal" or "Anomaly" (as found in the raw file)
        - is_anomaly (int): 1 if label == "Anomaly", else 0

    These labels are used only for evaluation, never as a model input
    feature, per the project's unsupervised-training requirement.
    """
    df = pd.read_csv(ANOMALY_LABEL_PATH)
    df = df.rename(columns={"BlockId": "block_id", "Label": "label"})
    df["is_anomaly"] = (df["label"] == "Anomaly").astype(int)
    return df


def load_event_templates() -> pd.DataFrame:
    """Load the dataset-provided event template table (EventId, EventTemplate)."""
    return pd.read_csv(EVENT_TEMPLATES_PATH)


def inspect_dataset() -> None:
    """Print a structured inspection report of the raw dataset files.

    Covers: filenames, dimensions, columns, sample records, missing values,
    duplicate records, anomaly label format, and event template count.
    """
    print(f"Dataset directory: {DATA_RAW_DIR}")
    missing = check_dataset_files()
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path.name}")
        return

    print(f"Files found: {[p.name for p in DATA_RAW_DIR.iterdir() if p.is_file()]}")

    labels = load_anomaly_labels()
    print("\n--- anomaly_label.csv ---")
    print(f"Dimensions: {labels.shape[0]} rows x {labels.shape[1]} columns")
    print(f"Columns: {list(labels.columns)}")
    print("Sample records:")
    print(labels.head(5).to_string(index=False))
    print(f"Missing values:\n{labels.isna().sum().to_string()}")
    print(f"Duplicate block ids: {labels['block_id'].duplicated().sum()}")
    print(f"Label value counts:\n{labels['label'].value_counts().to_string()}")

    if EVENT_TEMPLATES_PATH.is_file():
        templates = load_event_templates()
        print("\n--- HDFS.log_templates.csv ---")
        print(f"Dimensions: {templates.shape[0]} rows x {templates.shape[1]} columns")
        print(f"Columns: {list(templates.columns)}")
        print(templates.to_string(index=False))

    with open(HDFS_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        sample_lines = [next(f) for _ in range(5)]
    print("\n--- HDFS.log (raw) ---")
    print("Sample records:")
    for line in sample_lines:
        print(f"  {line.rstrip()}")


if __name__ == "__main__":
    inspect_dataset()
