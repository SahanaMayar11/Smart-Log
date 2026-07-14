"""Build a per-block behavioral feature matrix from parsed HDFS log records.

One row = one HDFS Block ID's full log sequence. Features are computed with
a single streaming pass over the ~1.5GB raw log file (src/log_parser.py),
accumulating per-block statistics incrementally rather than materializing
every parsed line in memory.

Feature documentation (why each feature may help anomaly detection):

- total_events (sequence length): Total number of log lines for the block.
  Anomalous blocks (e.g. blocks that never complete replication, or that
  are repeatedly retried) tend to have unusually short or unusually long
  sequences compared with a normal write/replicate/close lifecycle.
- unique_events: Count of distinct event templates (of E1-E29) seen for the
  block. A very low or very high diversity relative to sequence length can
  indicate an incomplete or looping lifecycle.
- event_diversity: unique_events / total_events. Normalizes unique_events by
  sequence length so short and long sequences are comparable.
- error_event_count: Count of events drawn from templates that represent
  exceptions, interruptions, or failures (E4, E7, E8, E10, E12, E14, E17,
  E20). Directly reflects operational failures during the block's
  lifecycle.
- warn_event_count / info_event_count: Count of WARN / INFO level lines.
  A high WARN share is a direct behavioral signal of abnormal handling.
- event_transitions: Number of times the event template changes between
  consecutive log lines for the block. Captures how "varied" the block's
  behavior sequence is, independent of which events occur.
- repeated_event_count: Number of consecutive log lines with the *same*
  event template (total_events - 1 - event_transitions). A block stuck
  repeating one event (e.g. repeated retries) is behaviorally distinct from
  one that progresses through varied events.
- time_span_seconds: Time between the block's first and last log line.
  Anomalously long-lived or unusually short block lifecycles can indicate
  problems (e.g. stalled replication vs. instant failure).
- avg_inter_event_seconds: time_span_seconds / (total_events - 1). Captures
  the pace of activity, independent of how many events occurred.
- distinct_components: Number of distinct HDFS components (e.g.
  DataXceiver, PacketResponder, FSNamesystem) that logged about this block.
  A block touched by an unusual number of subsystems can indicate abnormal
  handling paths.
- event_freq_<EventId> (one column per E1-E29 and E_UNKNOWN): Fraction of
  the block's events that belong to each event template
  (count of that template / total_events). This is the "event-template
  frequency" behavioral fingerprint of the block, normalized by sequence
  length so it is comparable across blocks with different numbers of
  events.

Login frequency, failed login attempts, and session duration are
intentionally not used: those concepts do not exist in the HDFS dataset,
which has no authentication events.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.log_parser import _EVENT_RULES, UNKNOWN_EVENT_ID, iter_log_records

ALL_EVENT_IDS = [event_id for event_id, _ in _EVENT_RULES] + [UNKNOWN_EVENT_ID]

# Event templates whose fixed text represents an exception, interruption, or
# failure condition (inspected directly from HDFS.log_templates.csv).
ERROR_EVENT_IDS = {"E4", "E7", "E8", "E10", "E12", "E14", "E17", "E20"}


def _parse_timestamp(date: str, time: str) -> int:
    """Convert a raw 'YYMMDD' + 'HHMMSS' pair into an integer second offset.

    Returns an integer that is monotonically comparable within the dataset's
    single-year span (the raw log has no explicit year, so a full datetime
    is unnecessary; only relative ordering/duration matters for features).
    """
    yy, mm, dd = int(date[0:2]), int(date[2:4]), int(date[4:6])
    hh, mi, ss = int(time[0:2]), int(time[2:4]), int(time[4:6])
    day_index = (yy * 372) + (mm * 31) + dd  # coarse but monotonic day index
    return day_index * 86400 + hh * 3600 + mi * 60 + ss


class _BlockAccumulator:
    """Accumulates streaming statistics for a single HDFS block."""

    __slots__ = (
        "total_events",
        "event_counts",
        "warn_count",
        "info_count",
        "components",
        "first_ts",
        "last_ts",
        "last_event_id",
        "transitions",
    )

    def __init__(self) -> None:
        self.total_events = 0
        self.event_counts: dict[str, int] = defaultdict(int)
        self.warn_count = 0
        self.info_count = 0
        self.components: set[str] = set()
        self.first_ts: int | None = None
        self.last_ts: int | None = None
        self.last_event_id: str | None = None
        self.transitions = 0

    def add(self, record) -> None:
        self.total_events += 1
        self.event_counts[record.event_id] += 1
        if record.level == "WARN":
            self.warn_count += 1
        elif record.level == "INFO":
            self.info_count += 1
        self.components.add(record.component)

        ts = _parse_timestamp(record.date, record.time)
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts

        if self.last_event_id is not None and self.last_event_id != record.event_id:
            self.transitions += 1
        self.last_event_id = record.event_id


def build_feature_matrix(log_path, progress_every: int = 2_000_000) -> pd.DataFrame:
    """Stream-parse the raw HDFS log and build the per-block feature matrix.

    Args:
        log_path: Path to the raw HDFS.log file.
        progress_every: Print a progress message every N parsed lines.

    Returns:
        A DataFrame indexed by nothing (block_id is a column) with one row
        per HDFS block and the behavioral features documented in this
        module's docstring.
    """
    accumulators: dict[str, _BlockAccumulator] = {}

    n_parsed = 0
    for record in iter_log_records(log_path):
        acc = accumulators.get(record.block_id)
        if acc is None:
            acc = _BlockAccumulator()
            accumulators[record.block_id] = acc
        acc.add(record)

        n_parsed += 1
        if progress_every and n_parsed % progress_every == 0:
            print(f"  ... parsed {n_parsed:,} log lines, {len(accumulators):,} blocks so far")

    print(f"  Finished parsing {n_parsed:,} log lines across {len(accumulators):,} blocks")

    rows = []
    for block_id, acc in accumulators.items():
        total = acc.total_events
        error_count = sum(acc.event_counts.get(eid, 0) for eid in ERROR_EVENT_IDS)
        time_span = (acc.last_ts - acc.first_ts) if acc.last_ts is not None else 0
        row = {
            "block_id": block_id,
            "total_events": total,
            "unique_events": len(acc.event_counts),
            "event_diversity": len(acc.event_counts) / total if total else 0.0,
            "error_event_count": error_count,
            "warn_event_count": acc.warn_count,
            "info_event_count": acc.info_count,
            "event_transitions": acc.transitions,
            "repeated_event_count": max(total - 1 - acc.transitions, 0),
            "time_span_seconds": time_span,
            "avg_inter_event_seconds": (time_span / (total - 1)) if total > 1 else 0.0,
            "distinct_components": len(acc.components),
        }
        for event_id in ALL_EVENT_IDS:
            row[f"event_freq_{event_id}"] = acc.event_counts.get(event_id, 0) / total if total else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the numerical feature column names (excludes identifiers/labels)."""
    exclude = {"block_id", "label", "is_anomaly"}
    return [c for c in df.columns if c not in exclude]


def clean_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN/infinite values in the feature matrix.

    Replaces +/-inf with NaN, then fills any remaining NaNs in numerical
    feature columns with 0.0 (a block with no observations of a given
    behavior legitimately has zero occurrences of it, so 0 is not an
    arbitrary fabricated value here, unlike e.g. imputing a missing income).
    """
    feature_cols = get_feature_columns(df)
    df = df.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    n_nan = int(df[feature_cols].isna().sum().sum())
    if n_nan:
        print(f"  Cleaning: filling {n_nan} NaN/inf feature values with 0.0")
        df[feature_cols] = df[feature_cols].fillna(0.0)
    return df
