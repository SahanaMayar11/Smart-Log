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

A Flask web app (`app.py`, workflow "Start application", port 5000) lets a
user upload a new HDFS log (plus an optional ground-truth labels CSV) and
run live inference against the already-trained models from `models/`
(`src/inference.py`) — no retraining. Each upload gets an isolated job id;
results (summary stats, model comparison if labels were provided, charts,
severity-filterable alerts table, CSV/JSON download) render on
`/jobs/<id>/results`. Uploaded raw files are deleted after scoring; job
results live in memory only (not persisted across app restarts).

## User preferences

- Follow the detailed project brief in
  `attached_assets/Pasted-Act-as-a-senior-Machine-Learning-Engineer-Python-Develo_1784024739351.txt`
  for scope, architecture, and constraints (e.g. don't fabricate dataset
  columns/results, keep code modular in `src/`, build phase by phase after
  inspecting the real dataset).
