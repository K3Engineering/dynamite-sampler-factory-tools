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
    csv_path TEXT,
    dwell_s REAL,
    guard_s REAL,
    namespace TEXT NOT NULL,  -- factory/user (cal group target)
    tx_power_dbm INTEGER,     -- DUT BLE TX power during the sweep (measurement condition)
    noise_group TEXT,         -- noise budget row used by the gates (pro/lite/override)
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
    namespace TEXT NOT NULL,  -- factory/user (cal group written)
    keys_written TEXT NOT NULL,  -- JSON object: {factory key: value}
    delta TEXT,                  -- JSON {dut_ch: [new-prior counts]}, user runs only
    script TEXT NOT NULL,
    run_id INTEGER REFERENCES runs(id)
)
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the factory log, creating the schema as needed.

    New columns are added to an existing log in place (the runs from before a
    column existed keep NULL for it), so a factory log written by an older
    tool keeps working after an upgrade."""
    con = sqlite3.connect(db_path)
    con.execute(_SCHEMA_RUNS)
    con.execute(_SCHEMA_SEGMENTS)
    con.execute(_SCHEMA_CALIBRATIONS)
    _add_missing_columns(con, "runs", {"tx_power_dbm": "INTEGER", "noise_group": "TEXT"})
    con.commit()
    return con


def _add_missing_columns(con, table, columns):
    have = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns.items():
        if name not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def insert_run(con: sqlite3.Connection, record: dict) -> int:
    """Open a run row (result 'running'). Returns the run id."""
    cur = con.execute(
        "INSERT INTO runs"
        " (ts_utc, device_address, device_name, board_model, firmware_rev,"
        "  cal_board_id, cal_board_uid, cal_port,"
        "  csv_path, dwell_s, guard_s, namespace, tx_power_dbm, noise_group, script)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["ts_utc"],
            record["device_address"],
            record.get("device_name"),
            record.get("board_model"),
            record.get("firmware_rev"),
            record.get("cal_board_id"),
            record.get("cal_board_uid"),
            record.get("cal_port"),
            record.get("csv_path"),
            record.get("dwell_s"),
            record.get("guard_s"),
            record["namespace"],
            record.get("tx_power_dbm"),
            record.get("noise_group"),
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
        "  namespace, keys_written, delta, script, run_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record["ts_utc"],
            record["device_address"],
            record.get("device_name"),
            record.get("board_model"),
            record.get("firmware_rev"),
            record["namespace"],
            json.dumps(record["keys_written"]),
            None if record.get("delta") is None else json.dumps(record["delta"]),
            record["script"],
            record.get("run_id"),
        ),
    )
    con.commit()
    return cur.lastrowid
