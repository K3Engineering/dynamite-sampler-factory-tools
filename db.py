"""sqlite factory log: one row per calibration run, one row per sweep
segment, plus the calibrations table (keys written) from the original mock.

The bulk stream lives in CSV capture files; the
DB holds run metadata, per-segment statistics, gate results,
and the keys written to the device.
"""

import json
import sqlite3
from pathlib import Path

_SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    device_address TEXT NOT NULL,
    device_name TEXT,
    board_model TEXT,
    firmware_rev TEXT,
    cal_board_id TEXT,
    cal_board_uid TEXT,     -- TMP118 48-bit unique ID (hex): physical board identity
    cal_port TEXT,
    exc_mv REAL,            -- optional manual DMM entry (cal.exc.mv)
    csv_path TEXT,
    dwell_s REAL,
    guard_s REAL,
    result TEXT NOT NULL DEFAULT 'running',  -- running/pass/fail/dry-run/aborted
    fail_reason TEXT,
    script TEXT NOT NULL
)
"""

_SCHEMA_SEGMENTS = """
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    phase INTEGER NOT NULL,
    seq_idx INTEGER NOT NULL,
    commanded_mv INTEGER NOT NULL,
    cal_channels TEXT NOT NULL,  -- "1,3"
    confirm TEXT,                -- cal board stdout echoing this command
    t_cmd_unix REAL NOT NULL,    -- host time when the command was issued
    ssn_start INTEGER,           -- valid window [ssn_start, ssn_end], unwrapped
    ssn_end INTEGER,
    n_samples INTEGER,
    missing INTEGER,             -- emitted-but-not-received samples in window
    temp_calboard_c REAL NOT NULL,  -- TMP118 read at the end of this segment
    means TEXT,                  -- JSON list of 4 per-channel means (null if empty)
    stds TEXT                    -- JSON list of 4 per-channel sample stds
)
"""

_SCHEMA_CALIBRATIONS = """
CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    device_address TEXT NOT NULL,
    device_name TEXT,
    board_model TEXT,
    firmware_rev TEXT,
    keys_written TEXT NOT NULL,  -- JSON object: {factory key: value}
    script TEXT NOT NULL
)
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the factory log, creating the schema as needed."""
    con = sqlite3.connect(db_path)
    con.execute(_SCHEMA_RUNS)
    con.execute(_SCHEMA_SEGMENTS)
    con.execute(_SCHEMA_CALIBRATIONS)
    con.commit()
    return con


def insert_run(con: sqlite3.Connection, record: dict) -> int:
    """Open a run row (result 'running'). Returns the run id."""
    cur = con.execute(
        "INSERT INTO runs"
        " (ts_utc, device_address, device_name, board_model, firmware_rev,"
        "  cal_board_id, cal_board_uid, cal_port, exc_mv,"
        "  csv_path, dwell_s, guard_s, script)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["ts_utc"],
            record["device_address"],
            record.get("device_name"),
            record.get("board_model"),
            record.get("firmware_rev"),
            record.get("cal_board_id"),
            record.get("cal_board_uid"),
            record.get("cal_port"),
            record.get("exc_mv"),
            record.get("csv_path"),
            record.get("dwell_s"),
            record.get("guard_s"),
            record["script"],
        ),
    )
    con.commit()
    return cur.lastrowid


def insert_segment(
    con: sqlite3.Connection, run_id: int, seg, confirm: str | None = None
) -> None:
    """Append one sweep segment (a cal_math.SegmentResult or anything with
    matching attributes). confirm is the cal board's stdout echo of the
    command, if captured."""
    con.execute(
        "INSERT INTO segments"
        " (run_id, phase, seq_idx, commanded_mv, cal_channels, confirm,"
        "  t_cmd_unix, ssn_start, ssn_end, n_samples, missing, temp_calboard_c,"
        "  means, stds)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            seg.phase,
            seg.seq_idx,
            seg.commanded_mv,
            ",".join(str(c) for c in seg.cal_channels),
            confirm,
            seg.t_cmd_unix,
            seg.ssn_start,
            seg.ssn_end,
            seg.n_samples,
            seg.missing,
            seg.temp_c,
            json.dumps(list(seg.means)),
            json.dumps(list(seg.stds)),
        ),
    )
    con.commit()


def finish_run(
    con: sqlite3.Connection, run_id: int, result: str, fail_reason: str | None = None
) -> None:
    con.execute(
        "UPDATE runs SET result = ?, fail_reason = ? WHERE id = ?",
        (result, fail_reason, run_id),
    )
    con.commit()


def log_calibration(con: sqlite3.Connection, record: dict) -> int:
    """Append a keys-written record. Returns the row id."""
    cur = con.execute(
        "INSERT INTO calibrations"
        " (ts_utc, device_address, device_name, board_model, firmware_rev,"
        "  keys_written, script, run_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["ts_utc"],
            record["device_address"],
            record.get("device_name"),
            record.get("board_model"),
            record.get("firmware_rev"),
            json.dumps(record["keys_written"]),
            record["script"],
            record.get("run_id"),
        ),
    )
    con.commit()
    return cur.lastrowid
