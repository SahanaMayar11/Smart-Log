"""Smart-Log: Automated Syslog Anomaly Detection for Server Security.

Entry point for the full pipeline. Currently this checks that the required
HDFS dataset files are present in data/raw/ before the pipeline (log parsing,
feature engineering, model training, SHAP explanation, alerting, evaluation)
can be implemented and executed, since implementation must be based on the
actual structure of the dataset rather than assumptions about it.
"""

from pathlib import Path

DATA_RAW_DIR = Path(__file__).parent / "data" / "raw"

REQUIRED_FILES = [
    "HDFS.log",
    "anomaly_label.csv",
]


def check_dataset_present() -> bool:
    """Check whether the required raw HDFS dataset files exist.

    Returns:
        True if all required files are present in data/raw/, False otherwise.
    """
    missing = [name for name in REQUIRED_FILES if not (DATA_RAW_DIR / name).is_file()]
    if missing:
        print("SMART-LOG: Required dataset files are missing from data/raw/:")
        for name in missing:
            print(f"  - {name}")
        print(
            "\nPlace the HDFS Log Anomaly Dataset files listed above into "
            f"{DATA_RAW_DIR} and re-run this script.\n"
            "Source: the loghub HDFS_1 dataset "
            "(https://github.com/logpai/loghub) provides HDFS.log and "
            "anomaly_label.csv."
        )
        return False
    return True


def main() -> None:
    """Run the Smart-Log pipeline."""
    if not check_dataset_present():
        return

    # Pipeline stages (log ingestion, parsing, preprocessing, feature
    # engineering, Isolation Forest, One-Class SVM, hybrid validation, SHAP
    # explainability, alert generation, evaluation) will be implemented in
    # src/ once the dataset structure has been inspected.
    print("Dataset files found. Pipeline implementation is next.")


if __name__ == "__main__":
    main()
