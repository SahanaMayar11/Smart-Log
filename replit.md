# Smart-Log

## Overview

Smart-Log is a behavior-based anomaly detection system for HDFS server logs
(Isolation Forest + One-Class SVM hybrid detection, SHAP explainability,
alert generation, and evaluation). See `README.md` for the full architecture,
dataset requirements, and how to run it.

## Current state

The full pipeline is implemented and runs end-to-end against the real
dataset (`data/raw/HDFS.log`, `anomaly_label.csv`, `HDFS.log_templates.csv`):
log parsing, feature engineering, Isolation Forest + One-Class SVM training,
hybrid AND/OR strategy comparison, SHAP explainability, alert generation,
evaluation metrics, and visualizations. Run `python main.py` (parsed
features are cached to `data/processed/block_features_raw.parquet` after the
first run, so subsequent runs skip the ~2.5-minute raw log re-parse). See
`README.md` for full architecture, feature documentation, and the latest
run's results table.

## User preferences

- Follow the detailed project brief in
  `attached_assets/Pasted-Act-as-a-senior-Machine-Learning-Engineer-Python-Develo_1784024739351.txt`
  for scope, architecture, and constraints (e.g. don't fabricate dataset
  columns/results, keep code modular in `src/`, build phase by phase after
  inspecting the real dataset).
