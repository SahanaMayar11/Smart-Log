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

- `HDFS.log` — 11,175,629 raw HDFS log lines (~1.5GB), one line per event in
  the format `<date> <time> <pid> <level> <component>: <message>`. Every
  line references exactly one HDFS Block ID (`blk_<signed integer>`),
  verified across the full file.
- `anomaly_label.csv` — 575,061 ground-truth per-block labels
  (558,223 Normal / 16,838 Anomaly), used only for evaluation and for
  selecting the training split, never as a model input feature.
- `HDFS.log_templates.csv` — the dataset's own 29 event templates (E1-E29),
  used to classify each parsed log message without inventing event types.

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

The full pipeline is implemented and has been run end-to-end against the
real dataset: log parsing, feature engineering, model training, hybrid
strategy comparison, SHAP explainability, alert generation, evaluation, and
visualizations. See Results below for the most recent run's numbers.

## Training Strategy

Isolation Forest and One-Class SVM are novelty-detection models: they are
fit on data assumed to represent normal behavior, then flag deviations.
Ground-truth labels are used only to decide *which rows* go into the
training split (never as a model input feature): 80% of the Normal-labeled
blocks form the training set; the remaining Normal blocks plus every
Anomaly-labeled block form the held-out test set used for all evaluation.
This avoids label leakage while still letting an unsupervised model be
evaluated meaningfully. One-Class SVM's RBF kernel has O(n^2)-O(n^3) training
cost, so it is fit on a random 10,000-row subsample of the training split
(documented in `src/one_class_svm_model.py`); Isolation Forest, which scales
much better, is trained on the full ~446,000-row training split.

## ML Models

Isolation Forest (primary detector, 200 trees) and One-Class SVM (secondary
validator, RBF kernel), combined via two hybrid strategies: **AND** (both
models must agree) and **OR** (either model flags it). All four
strategies/models are evaluated independently; the best by F1-Score is used
to drive alert generation.

## Feature Engineering

41 behavioral features are computed per HDFS Block ID from a single
streaming pass over the raw log (see `src/feature_engineering.py` for full
documentation of each feature and its rationale): sequence length, unique
event count, event diversity, error/warning/info event counts, event
transition and repetition counts, time span and average inter-event timing,
distinct component count, and the normalized frequency of each of the 29
dataset-defined event templates (E1-E29) plus an "unknown template"
fallback bucket.

## SHAP Explainability

`shap.TreeExplainer` (verified compatible with scikit-learn's
`IsolationForest`) explains every anomaly the Isolation Forest flags. For
each flagged block, the 3 features with the most negative SHAP values (the
strongest push toward "anomalous") are surfaced in a human-readable
explanation sentence and in the alert record.

## Alert Generation

Each alert includes an alert ID, block ID, anomaly score, severity
(LOW/MEDIUM/HIGH/CRITICAL, from project-defined thresholds on the min-max
normalized anomaly score across the current run's flagged alerts), the top
3 contributing features, and a human-readable explanation, saved to
`outputs/alerts/` as both CSV and JSON.

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, and Confusion Matrix are computed
against the ground-truth `anomaly_label.csv` on the held-out test set,
separately for Isolation Forest, One-Class SVM, Hybrid AND, and Hybrid OR.

## Results (most recent run)

Test set: 128,483 blocks (111,645 Normal / 16,838 Anomaly).

| Model            | Accuracy | Precision | Recall | F1     |
|------------------|----------|-----------|--------|--------|
| Isolation Forest | 0.9235   | 0.7563    | 0.6138 | 0.6777 |
| One-Class SVM    | 0.9582   | 0.7593    | 0.9969 | 0.8620 |
| Hybrid AND       | 0.9310   | 0.8137    | 0.6138 | 0.6998 |
| Hybrid OR        | 0.9507   | 0.7276    | 0.9969 | 0.8412 |

Best strategy by F1-Score: **One-Class SVM**, which generated 22,106
predicted anomalies on the test set. Isolation Forest's flagged anomalies
(used for alert generation, since it is the designated primary detector)
produced 13,666 alerts. These numbers will vary slightly between runs where
randomness affects the training subsample, but `random_state=42` is used
everywhere sklearn/numpy accept a seed to keep results reproducible.

## Output Files

- `models/` — persisted trained models and scaler (joblib).
- `outputs/plots/` — generated visualizations.
- `outputs/alerts/` — generated alerts (CSV/JSON).
- `outputs/metrics/` — evaluation results.
