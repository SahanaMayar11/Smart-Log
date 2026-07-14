# Smart-Log

## Overview

Smart-Log is a behavior-based anomaly detection system for HDFS server logs
(Isolation Forest + One-Class SVM hybrid detection, SHAP explainability,
alert generation, and evaluation). See `README.md` for the full architecture,
dataset requirements, and how to run it.

## Current state

Environment is set up: Python 3.11 with pandas, numpy, scikit-learn,
matplotlib, seaborn, shap, and joblib installed (see `requirements.txt`), and
the project directory structure (`data/`, `models/`, `outputs/`, `src/`) is
scaffolded. The dataset (`HDFS.log`, `anomaly_label.csv`) is not yet present
in `data/raw/` — the actual pipeline (parsing, feature engineering, model
training, SHAP, alerting, evaluation) must be implemented against the real
file structure, so it has not been written yet. Run `python main.py` to check
dataset presence.

## User preferences

- Follow the detailed project brief in
  `attached_assets/Pasted-Act-as-a-senior-Machine-Learning-Engineer-Python-Develo_1784024739351.txt`
  for scope, architecture, and constraints (e.g. don't fabricate dataset
  columns/results, keep code modular in `src/`, build phase by phase after
  inspecting the real dataset).
