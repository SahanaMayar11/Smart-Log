---
name: HDFS log anomaly detection pipeline design
description: Design decisions for parsing loghub-style HDFS logs and running unsupervised anomaly detection (Isolation Forest / One-Class SVM) without label leakage, at ~11M-line scale.
---

**Event template classification without a structured-log file.** The
loghub HDFS_1 dataset ships `HDFS.log_templates.csv` (29 fixed templates
with `[*]` wildcards) but not a pre-computed per-line EventId mapping for
the full raw log. Classify each parsed message by checking, in a fixed
order, whether all of a template's literal (non-wildcard) substrings are
present in the message — ordered so that templates sharing substrings
(e.g. three different "addStoredBlock" variants) are disambiguated by their
more specific text. This is far faster than compiling/matching 29 regexes
per line across 11M+ lines, and had 0 unmatched messages against the real
HDFS_1 log.

**Why:** Fabricating event IDs or writing dataset-specific parsing before
inspecting the real file risks silently wrong features; using the dataset's
own template list as ground truth for classification avoids that while
staying fast enough for full-file streaming (~2.5 min for 11.17M lines on a
single pass with a Python dict-based per-block accumulator, no full
DataFrame materialization of raw lines).

**Semi-supervised train/test split to avoid label leakage in unsupervised
models.** For per-entity behavioral anomaly detection (e.g. one row per
HDFS Block ID) where labels exist only for evaluation: use the labels to
choose *which rows* go into the training split (e.g. 80% of Normal-labeled
entities), never as a model input feature. Train Isolation Forest /
One-Class SVM only on that assumed-normal split; evaluate on the remaining
Normal rows plus all Anomaly rows. This is standard for novelty-detection
benchmarks and keeps sklearn's contamination/nu-style unsupervised models
genuinely unsupervised at fit time.

**Why:** These models are literally novelty detectors that need to see
"normal" during fit; letting a small labeled slice guide the split (not the
fit) is the accepted way to benchmark them without leaking labels into
training.

**One-Class SVM (RBF) does not scale to hundreds of thousands of rows.**
Its training cost is O(n^2)-O(n^3). Fit it on a fixed random subsample
(e.g. 10k rows) of the training split even when Isolation Forest is trained
on the full split; document this as a documented engineering tradeoff.
Evaluation still runs on the full held-out test set, so this only affects
training time, not what gets measured.
