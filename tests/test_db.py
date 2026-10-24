import sqlite3

import db


def test_schema_creates_and_roundtrips(tmp_path):
    con = db.connect(tmp_path / "log.db")
    run_id = db.insert_run(
        con,
        {
            "ts_utc": "2026-10-22T00:00:00+00:00",
            "device_address": "AA",
            "namespace": "factory",
            "script": "t",
            "tx_power_dbm": -12,
            "noise_group": "lite",
        },
    )
    row = con.execute(
        "SELECT tx_power_dbm, noise_group FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row == (-12, "lite")


def test_older_log_gains_the_new_columns(tmp_path):
    """A factory log written before tx_power_dbm/noise_group existed must open
    and take new rows; old rows keep NULL."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL,"
        " device_address TEXT NOT NULL, device_name TEXT, board_model TEXT,"
        " firmware_rev TEXT, cal_board_id TEXT, cal_board_uid TEXT, cal_port TEXT,"
        " csv_path TEXT, dwell_s REAL, guard_s REAL, namespace TEXT NOT NULL,"
        " result TEXT NOT NULL DEFAULT 'running', fail_reason TEXT,"
        " script TEXT NOT NULL)"
    )
    con.execute(
        "INSERT INTO runs (ts_utc, device_address, namespace, script)"
        " VALUES ('2026-08-17T00:00:00+00:00','AA','factory','old')"
    )
    con.commit()
    con.close()

    con = db.connect(path)
    columns = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
    assert {"tx_power_dbm", "noise_group"} <= columns
    assert con.execute("SELECT tx_power_dbm FROM runs WHERE id = 1").fetchone() == (
        None,
    )
    run_id = db.insert_run(
        con,
        {
            "ts_utc": "2026-10-22T00:00:00+00:00",
            "device_address": "BB",
            "namespace": "factory",
            "script": "t",
            "tx_power_dbm": -24,
            "noise_group": "pro",
        },
    )
    assert con.execute(
        "SELECT tx_power_dbm, noise_group FROM runs WHERE id = ?", (run_id,)
    ).fetchone() == (-24, "pro")
