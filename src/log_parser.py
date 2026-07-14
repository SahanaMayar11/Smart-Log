"""Parse raw HDFS log lines into structured records.

The raw HDFS log (data/raw/HDFS.log) has one record per line in a fixed,
space-delimited format, confirmed by direct inspection of the dataset:

    <date> <time> <pid> <level> <component>: <message>

Example:
    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block \
blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010

Every line in the dataset references exactly one HDFS Block ID (verified by
scanning the full 11,175,629-line file: 0 lines have zero or multiple
`blk_*` ids), so each line is parsed and attached to a single block.

Event templates (E1-E29) come from the dataset's own
`HDFS.log_templates.csv` (loghub HDFS_1), which lists the fixed message
template for each event, with `[*]` marking variable segments. Event IDs are
therefore not invented: they are assigned by matching each parsed message
against these 29 known templates. Each template's fixed (non-`[*]`) text
segments are checked as substrings of the message; templates are ordered so
that shared substrings (e.g. the three "addStoredBlock" variants, or
"Receiving"/"Received" pairs) are disambiguated correctly. Every message in
the dataset matches exactly one of the 29 known templates in practice; any
message that matches none is labeled "E_UNKNOWN" rather than assigned a
fabricated template.
"""

import re
from dataclasses import dataclass

BLOCK_ID_PATTERN = re.compile(r"blk_-?\d+")

# (event_id, list of required substrings) - order matters: more specific
# templates (subsets that share substrings with others) are checked first.
_EVENT_RULES = [
    ("E1", ["Adding an already existing block"]),
    ("E2", ["Verification succeeded for"]),
    ("E4", ["Got exception while serving"]),
    ("E3", ["Served block", " to "]),
    ("E6", ["Received block", "src:", "dest:", "of size"]),
    ("E5", ["Receiving block", "src:", "dest:"]),
    ("E7", ["writeBlock", "received exception"]),
    ("E8", ["PacketResponder", "for block", "Interrupted"]),
    ("E9", ["Received block", "of size", "from"]),
    ("E10", ["PacketResponder", "Exception"]),
    ("E11", ["PacketResponder", "for block", "terminating"]),
    ("E12", ["Exception writing block", "to mirror"]),
    ("E13", ["Receiving empty packet for block"]),
    ("E14", ["Exception in receiveBlock for block"]),
    ("E15", ["Changing block file offset of block"]),
    ("E16", ["Transmitted block", " to "]),
    ("E17", ["Failed to transfer", " to ", "got"]),
    ("E18", ["Starting thread to transfer block", " to "]),
    ("E19", ["Reopen Block"]),
    ("E20", ["Unexpected error trying to delete block", "BlockInfo not found in volumeMap"]),
    ("E21", ["Deleting block", "file"]),
    ("E22", ["NameSystem.allocateBlock"]),
    ("E23", ["NameSystem.delete", "is added to invalidSet of"]),
    ("E24", ["Removing block", "neededReplications"]),
    ("E25", ["BLOCK* ask", "to replicate"]),
    ("E27", ["Redundant addStoredBlock request received for"]),
    ("E28", ["addStoredBlock request received for", "does not belong to any file"]),
    ("E26", ["addStoredBlock: blockMap updated"]),
    ("E29", ["PendingReplicationMonitor timed out block"]),
]

UNKNOWN_EVENT_ID = "E_UNKNOWN"


@dataclass(frozen=True)
class LogRecord:
    """A single structured HDFS log record."""

    date: str
    time: str
    pid: str
    level: str
    component: str
    message: str
    block_id: str
    event_id: str


def classify_event(message: str) -> str:
    """Assign an event template id (E1-E29) to a log message.

    Matches the message against the fixed text segments of the 29 event
    templates documented in data/raw/HDFS.log_templates.csv. Returns
    "E_UNKNOWN" if no known template matches, rather than fabricating one.
    """
    for event_id, required_substrings in _EVENT_RULES:
        if all(substr in message for substr in required_substrings):
            return event_id
    return UNKNOWN_EVENT_ID


def parse_line(line: str) -> LogRecord | None:
    """Parse a single raw HDFS log line into a LogRecord.

    Returns None if the line does not match the expected
    "<date> <time> <pid> <level> <component>: <message>" structure or does
    not reference exactly one block id (both occur 0 times in the actual
    dataset, but are handled explicitly rather than assumed away).
    """
    parts = line.rstrip("\n").split(" ", 5)
    if len(parts) != 6:
        return None
    date, time, pid, level, component, message = parts
    if not component.endswith(":"):
        return None
    component = component[:-1]

    block_ids = set(BLOCK_ID_PATTERN.findall(message))
    if len(block_ids) != 1:
        return None
    block_id = next(iter(block_ids))

    event_id = classify_event(message)
    return LogRecord(
        date=date,
        time=time,
        pid=pid,
        level=level,
        component=component,
        message=message,
        block_id=block_id,
        event_id=event_id,
    )


def iter_log_records(log_path):
    """Stream-parse every line of the raw HDFS log file.

    Yields LogRecord objects one at a time so the ~1.5GB raw log file never
    needs to be fully materialized in memory.
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            record = parse_line(line)
            if record is not None:
                yield record
