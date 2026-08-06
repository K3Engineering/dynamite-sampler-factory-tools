"""Factory script 2 of 2: calibration run (currently a dummy) + factory log.

Connects to the DUT via the python-api, performs a placeholder
"calibration" (no stimulus for now; the results are just the model nominals),
writes them to the Factory namespace (overwriting flash_factory_nominals's
stuff), and appends a record to the sqlite factory log.

Usage:
    python run_calibration.py [--address AA:BB:CC:DD:EE:FF] [--board v700P] [--db factory_log.db]
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from dut_kvs import FOLDER_FACTORY, DutKvs, KvsError
from nominal_values import BOARD_MODELS

# For DeviceInfo reads via the python-api (dut_kvs put it on sys.path).
import dynamite_sampler_api as ds
from dynamite_sampler_bleak_util import read_characteristic

SCRIPT_VERSION = "run_calibration 0.1"
PROVENANCE_DUMMY = "dummycal"

DEFAULT_DB_PATH = Path(__file__).resolve().with_name("factory_log.db")

_SCHEMA = """
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


def log_calibration(db_path: Path, record: dict) -> int:
    """Append a calibration record to the factory log. Returns the row id."""
    with sqlite3.connect(db_path) as con:
        con.execute(_SCHEMA)
        cur = con.execute(
            "INSERT INTO calibrations"
            " (ts_utc, device_address, device_name, board_model, firmware_rev,"
            "  keys_written, script)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record["ts_utc"],
                record["device_address"],
                record["device_name"],
                record["board_model"],
                record["firmware_rev"],
                json.dumps(record["keys_written"]),
                record["script"],
            ),
        )
        return cur.lastrowid


def dummy_calibrate(board_model: str | None) -> dict[str, str]:
    """Placeholder for the real 5-point sweep. "Measured" values are the
    model nominals, tagged with a dummy provenance."""
    nominals = BOARD_MODELS.get(board_model or "")
    if nominals is None:
        print(
            f"warning: board model {board_model!r} unknown, writing placeholder zeros"
        )
        nominals = {
            "adc_fsr": "0.0",
            "adc_gain": "0.0",
            "exc": "0.0",
            "afe_gain": "0.0",
        }
    entries = {key: f"{value},{PROVENANCE_DUMMY}" for key, value in nominals.items()}
    entries["cal_date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return entries


async def run(args: argparse.Namespace) -> int:
    async with await DutKvs.connect(args.address) as dut:
        firmware_rev = await read_characteristic(
            dut.client, ds.DeviceInfo.FirmwareRevision
        )

        # The board model provisioned by script 1, unless overridden.
        board_model = args.board
        if board_model is None:
            try:
                board_model = await dut.get(FOLDER_FACTORY, "board_model")
            except KvsError:
                print("warning: board_model not provisioned and --board not given")

        cal_entries = dummy_calibrate(board_model)

        failures = 0
        for key, value in cal_entries.items():
            await dut.set(FOLDER_FACTORY, key, value)
            readback = await dut.get(FOLDER_FACTORY, key)
            ok = readback == value
            failures += not ok
            print(
                f"  {key:12s} = {value:32s} {'ok' if ok else f'MISMATCH (read {readback!r})'}"
            )
        if failures:
            print(f"FAILED: {failures} of {len(cal_entries)} keys did not read back")
            return 1

        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device_address": dut.client.address,
            "device_name": dut.device_name,
            "board_model": board_model,
            "firmware_rev": firmware_rev,
            "keys_written": cal_entries,
            "script": SCRIPT_VERSION,
        }
        row_id = log_calibration(args.db, record)
        print(f"Logged calibration record #{row_id} to {args.db}")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        help="BLE address of the DUT (default: auto-detect, only one may be in range)",
    )
    parser.add_argument(
        "--board",
        choices=sorted(BOARD_MODELS),
        help="board model (default: read from the provisioned Factory namespace)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="sqlite factory log path (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(run(args)))
    except KvsError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
