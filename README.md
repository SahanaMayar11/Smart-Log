# Smart-Log: Automated Syslog Anomaly Detection for Server Security

## Project Overview

Smart-Log is a behavior-based anomaly detection system for HDFS server logs.
It parses raw HDFS log records, extracts behavioral features per log block,
and detects anomalous behavior using an Isolation Forest model validated by a
One-Class SVM, with SHAP-based explanations and generated security alerts.

## Problem Statement

Server logs contain evidence of abnormal or potentially malicious behavior,
but manually reviewing large volumes of raw logs is impractical. Smart-Log
automates this by learning what normal HDFS block behavior looks like and
flagging deviations, without relying on the ground-truth anomaly labels
during training.

## Objectives

- Parse raw HDFS logs into structured, per-block log sequences.
- Engineer behavioral features describing each block's log activity.
- Detect anomalies with an unsupervised Isolation Forest model.
- Validate/refine detections with a One-Class SVM.
- Combine both models using experimentally-compared hybrid strategies.
- Explain each detected anomaly with SHAP feature attributions.
- Generate severity-ranked, human-readable security alerts.
- Evaluate all strategies against ground-truth labels using standard
  classification metrics.

## Proposed Architecture

```
HDFS Server Logs
      |
Log Ingestion
      |
Log Preprocessing (Parsing, Cleaning, Structuring)
      |
Feature Engineering
      |
Log Behavioral Feature Matrix
      |
Isolation Forest (Primary Anomaly Detection)
      |
One-Class SVM (Anomaly Validation / Refinement)
      |
SHAP Explainability
      |
Generated Alerts (Severity + Explanation)
      |
Evaluation Metrics (Accuracy, Precision, Recall, F1-Score)
```

## Dataset

This project uses the **HDFS Log Anomaly Dataset** (loghub `HDFS_1`), which
consists of:

- `HDFS.log` — raw HDFS log lines.
- `anomaly_label.csv` — ground-truth per-block anomaly labels, used only for
  evaluation, never for training the unsupervised models.

## Technologies Used

Python 3.11, Pandas, NumPy, scikit-learn, Regex, Matplotlib, Seaborn, SHAP,
Joblib.

## Project Structure

```
smart-log/
├── data/
│   ├── raw/           # place HDFS.log and anomaly_label.csv here
│   └── processed/
├── models/            # saved IsolationForest / OneClassSVM / scaler
├── outputs/
│   ├── alerts/        # generated alerts (CSV/JSON)
│   ├── plots/         # generated visualizations
│   └── metrics/        # evaluation metrics
├── src/
│   ├── data_loader.py
│   ├── log_parser.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── isolation_forest_model.py
│   ├── one_class_svm_model.py
│   ├── hybrid_detector.py
│   ├── shap_explainer.py
│   ├── alert_generator.py
│   ├── evaluator.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

Dependencies are already installed in this Replit environment. To install
them elsewhere:

```
pip install -r requirements.txt
```

## Dataset Placement

Download the HDFS_1 dataset (e.g. from the
[loghub](https://github.com/logpai/loghub) repository) and place these files
in `data/raw/`:

- `data/raw/HDFS.log`
- `data/raw/anomaly_label.csv`

## How to Run

```
python main.py
```

Running `main.py` before the dataset files are in place will print exactly
which files are missing and where to put them.

## Status

Project scaffolding, environment, and dependency installation are complete.
The log parsing, feature engineering, model training, SHAP explainability,
alert generation, and evaluation stages will be implemented after the actual
dataset files are inspected, since the parsing and feature logic must match
the real structure of `HDFS.log` and `anomaly_label.csv` rather than assumed
column layouts.

## ML Models

Once implemented: Isolation Forest (primary detector) and One-Class SVM
(secondary validator), combined via experimentally-compared hybrid strategies
(AND / OR).

## Feature Engineering

Behavioral features will be computed per HDFS Block ID (e.g. event
frequency, sequence length, unique/error/warning/info event counts, event
transition characteristics), documented in `src/feature_engineering.py` once
implemented against the real log format.

## SHAP Explainability

SHAP will be used to explain which behavioral features drove each anomaly
score, with human-readable, per-alert explanations.

## Alert Generation

Each alert will include an alert ID, block ID, anomaly prediction, anomaly
score, severity (LOW/MEDIUM/HIGH/CRITICAL, derived from normalized anomaly
scores), top contributing features, and a human-readable explanation, saved
to `outputs/alerts/` as CSV and JSON.

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, and Confusion Matrix will be computed
against the ground-truth `anomaly_label.csv`, separately for Isolation
Forest, One-Class SVM, Hybrid AND, and Hybrid OR.

## Output Files

- `models/` — persisted trained models and scaler (joblib).
- `outputs/plots/` — generated visualizations.
- `outputs/alerts/` — generated alerts (CSV/JSON).
- `outputs/metrics/` — evaluation results.
