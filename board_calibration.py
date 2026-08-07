"""Run the 5-point factory board calibration against the calibration board.

Flow:
  1. Connect to the DUT over BLE; read the provisioning (board model,
     nominals) and the ADC's PGA register readback.
  2. Connect to the calibration board (USB serial, raw REPL via mpremote).
  3. Stream the entire ADC feed to CSV while stepping the cal board through a
     zero-anchored reversal sweep in two phases — channels 1+3, then 2+4
     (channels sharing a divider bridge can't be driven simultaneously).
  4. Segment the capture per commanded state (guard + dwell around SSN
     windows), compute the five per-config means per channel, run the gates.
  5. On pass: write ch{i}.r / ch{i}.raw / cal.* to the Factory namespace
     (read-back verified) and log everything to the factory DB.
     On any gate failure: abort loudly, write nothing (DB + CSV remain).

Usage:
    python board_calibration.py [--address AA:BB:CC:DD:EE:FF] [--cal-port COM5]
                                [--dwell 1.0] [--guard 0.4] [--dry-run]
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kvs_api_shim import FOLDER_FACTORY, KvsClient, KvsError

import dynamite_sampler_api as ds  # noqa: E402
from dynamite_sampler_bleak_util import FeedSession, read_characteristic  # noqa: E402
from capture import FeedRecorder  # noqa: E402

import cal_math  # noqa: E402
import db  # noqa: E402
from calboard_driver import CalBoard, CalBoardError  # noqa: E402
from nominal_values import BOARD_MODELS  # noqa: E402

SCRIPT_VERSION = "board_calibration 1.0"

DEFAULT_DB_PATH = Path(__file__).resolve().with_name("factory_log.db")
DEFAULT_CAPTURE_DIR = Path(__file__).resolve().with_name("captures")

# Sweep phases in cal-board channels: one channel per bridge at a time.
# DUT channel index = cal channel - 1 (fixture wiring).
PHASES = ((1, 3), (2, 4))

# Grace after stopping the feed: KVS commands are rejected while the device
# streams (firmware device lock), and unsubscription takes a moment.
_KVS_WRITE_DELAY_S = 0.5


async def read_flash_nominals(dut: KvsClient) -> dict[str, float]:
    """The provisioned analog nominals from flash (provenance stripped).

    Values follow the '<number>[,provenance]' scalar grammar of
    docs/flash-schema-v1.md (repo root)."""
    out = {}
    for key in ("adc_fsr", "exc", "afe_gain"):
        try:
            raw = await dut.get(FOLDER_FACTORY, key)
        except KvsError as e:
            raise KvsError(
                f"{key!r} not provisioned — run flash_factory_nominals.py first ({e})"
            ) from e
        out[key] = float(raw.split(",")[0].strip())
    return out


async def set_verified(
    dut: KvsClient, key: str, value: str, attempts: int = 3
) -> str | None:
    """SET + read-back verify, with retries (device-lock grace). Returns the
    readback on success, None on mismatch; re-raises KvsError after retries."""
    for attempt in range(attempts):
        try:
            await dut.set(FOLDER_FACTORY, key, value)
            readback = await dut.get(FOLDER_FACTORY, key)
            return readback if readback == value else None
        except KvsError:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(_KVS_WRITE_DELAY_S)


async def run(args: argparse.Namespace) -> int:
    gate_params = cal_math.GateParams(
        min_window_samples=args.min_window_samples,
        max_missing=args.max_missing,
        max_std_counts=args.max_std_counts,
        min_gap_counts=args.min_gap_counts,
        pass_tol_counts=args.pass_tol_counts,
        zero_tol_counts=args.zero_tol_counts,
        span_tol=args.span_tol,
    )

    async with await KvsClient.connect(args.address) as dut:
        firmware_rev = await read_characteristic(
            dut.client, ds.DeviceInfo.FirmwareRevision
        )
        hw_rev = await read_characteristic(dut.client, ds.DeviceInfo.HardwareRevision)
        adc_config = await read_characteristic(dut.client, ds.DynamiteSampler.ADCConfig)
        if adc_config is None:
            raise KvsError("ADC config unreadable — cannot cross-check the span")

        board_model = args.board
        if board_model is None:
            try:
                board_model = await dut.get(FOLDER_FACTORY, "board_model")
            except KvsError as e:
                raise KvsError(
                    "board_model not provisioned — run flash_factory_nominals.py "
                    "first (or pass --board)"
                ) from e
        if board_model not in BOARD_MODELS:
            raise KvsError(f"unknown board model {board_model!r}")
        if hw_rev and hw_rev != board_model:
            raise KvsError(
                f"provisioned board_model {board_model!r} does not match the "
                f"firmware's Hardware Revision {hw_rev!r} — wrong unit or bad "
                "provisioning"
            )
        nominals = await read_flash_nominals(dut)

        cal = await asyncio.to_thread(CalBoard.connect, args.cal_port)
        print(f"Calibration board on {cal.port}: {cal.fw_id}")

        # Expected +/-FS span per channel, from the flash nominals and the
        # ADC's own PGA readback — the gross-error cross-check.
        expected_span = [
            cal_math.expected_span_counts(
                nominals["adc_fsr"], nominals["afe_gain"], pga, nominals["exc"]
            )
            for pga in adc_config.gains
        ]

        started = datetime.now(timezone.utc)
        safe_mac = "".join(c for c in dut.client.address if c.isalnum())
        csv_path = (
            args.capture_dir
            / started.strftime("%Y%m%d")
            / f"cal_{started:%Y%m%d_%H%M%S}_{safe_mac}.csv"
        )

        con = db.connect(args.db)
        run_id = db.insert_run(
            con,
            {
                "ts_utc": started.isoformat(timespec="seconds"),
                "device_address": dut.client.address,
                "device_name": dut.device_name,
                "board_model": board_model,
                "firmware_rev": firmware_rev,
                "cal_board_id": cal.fw_id,
                "cal_port": cal.port,
                "exc_mv": args.exc_mv,
                "csv_path": str(csv_path),
                "dwell_s": args.dwell,
                "guard_s": args.guard,
                "script": SCRIPT_VERSION,
            },
        )
        print(f"Run #{run_id}: capture -> {csv_path}")

        recorder = FeedRecorder(csv_path)
        session = FeedSession(
            dut.client,
            callbacks_feeddata=[recorder],
            device_info={"FirmwareRevision": firmware_rev, "ADCConfig": adc_config},
        )
        seg_meta = []
        try:
            await session.start()
            deadline = time.monotonic() + 10.0
            while recorder.last_ssn is None:
                if time.monotonic() > deadline:
                    raise KvsError("no ADC feed data within 10 s of subscribing")
                await asyncio.sleep(0.05)
            print(f"Feed running (first SSN {recorder.last_ssn}). Sweeping...")

            for phase_idx, (cha, chb) in enumerate(PHASES):
                for seq_idx, mv in enumerate(cal_math.SWEEP_SEQUENCE_MV):
                    t_cmd = time.time()
                    confirm = await asyncio.to_thread(
                        cal.set_channels, {cha: mv, chb: mv}
                    )
                    await asyncio.sleep(args.guard)
                    ssn_start = (recorder.last_ssn or 0) + 1
                    await asyncio.sleep(args.dwell)
                    ssn_end = recorder.last_ssn
                    seg_meta.append(
                        {
                            "phase": phase_idx,
                            "seq_idx": seq_idx,
                            "mv": mv,
                            "channels": (cha, chb),
                            "t_cmd": t_cmd,
                            "ssn_start": ssn_start,
                            "ssn_end": ssn_end,
                            "confirm": confirm,
                        }
                    )
                    print(
                        f"  phase {phase_idx} ch{cha}+{chb} {mv:+3d} mV "
                        f"(ssn {ssn_start}..{ssn_end})"
                    )

            # Stop the stream before KVS access (firmware device lock).
            await session.stop()
            await asyncio.sleep(_KVS_WRITE_DELAY_S)

            # Segment the capture into per-state windows and reduce.
            segments = []
            for meta in seg_meta:
                ssn_start, ssn_end = meta["ssn_start"], meta["ssn_end"]
                rows = (
                    recorder.window(ssn_start, ssn_end)
                    if ssn_end is not None and ssn_end >= ssn_start
                    else []
                )
                means, stds = cal_math.segment_stats(rows)
                expected = ssn_end - ssn_start + 1 if rows else 0
                seg = cal_math.SegmentResult(
                    phase=meta["phase"],
                    seq_idx=meta["seq_idx"],
                    commanded_mv=meta["mv"],
                    cal_channels=meta["channels"],
                    t_cmd_unix=meta["t_cmd"],
                    ssn_start=ssn_start,
                    ssn_end=ssn_end,
                    n_samples=len(rows),
                    missing=expected - len(rows),
                    means=means,
                    stds=stds,
                )
                segments.append(seg)
                db.insert_segment(con, run_id, seg, confirm=meta["confirm"])

            config_values = cal_math.collect_config_values(segments)
            failures = cal_math.gate_run(
                segments, config_values, expected_span, gate_params
            )
            if failures:
                db.finish_run(con, run_id, "fail", "; ".join(failures))
                print(f"CALIBRATION FAILED (run #{run_id}) — nothing written:")
                for failure in failures:
                    print(f"  - {failure}")
                return 1

            readings = cal_math.final_readings(config_values)
            entries = cal_math.build_flash_entries(
                readings, cal.fw_id, exc_mv=args.exc_mv, now=started
            )
            print("Gates passed. Calibration readings (raw counts, storage order):")
            for dut_ch in range(4):
                print(f"  ch{dut_ch}: {entries[f'ch{dut_ch}.raw']}")

            if args.dry_run:
                db.finish_run(con, run_id, "dry-run")
                print("Dry run — would write to the Factory namespace:")
                for key, value in entries.items():
                    print(f"  {key:12s} = {value}")
                return 0

            mismatches = 0
            for key, value in entries.items():
                readback = await set_verified(dut, key, value)
                ok = readback == value
                mismatches += not ok
                print(
                    f"  {key:12s} = {value:56s} "
                    f"{'ok' if ok else f'MISMATCH (read {readback!r})'}"
                )
            if mismatches:
                db.finish_run(con, run_id, "fail", f"{mismatches} keys misverified")
                print(f"FAILED: {mismatches} of {len(entries)} keys did not read back")
                return 1

            log_id = db.log_calibration(
                con,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "device_address": dut.client.address,
                    "device_name": dut.device_name,
                    "board_model": board_model,
                    "firmware_rev": firmware_rev,
                    "keys_written": entries,
                    "script": SCRIPT_VERSION,
                    "run_id": run_id,
                },
            )
            db.finish_run(con, run_id, "pass")
            print(f"Run #{run_id} PASS — calibration record #{log_id} in {args.db}")
            return 0
        except Exception as e:
            # Any mid-run failure leaves a fully interpretable trail.
            db.finish_run(con, run_id, "aborted", str(e))
            raise
        finally:
            await session.stop()  # idempotent
            cal.close()


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
        "--cal-port",
        help="serial port of the calibration board (default: probe plausible ports)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="sqlite factory log path (default: %(default)s)",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help="directory for raw stream captures (default: %(default)s)",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=1.0,
        help="measurement dwell per sweep state, seconds (default: %(default)s; "
        "the Allan minimum is ~0.2 s, longer buys nothing)",
    )
    parser.add_argument(
        "--guard",
        type=float,
        default=0.4,
        help="settling guard after each relay command, seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--exc-mv",
        type=float,
        help="DMM-measured excitation in mV (stored as cal.exc.mv; omit to skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="measure and gate, but write nothing to the device",
    )
    gates = parser.add_argument_group("gate thresholds (gross-fault defaults)")
    gates.add_argument("--min-window-samples", type=int, default=100)
    gates.add_argument("--max-missing", type=int, default=0)
    gates.add_argument("--max-std-counts", type=float, default=100.0)
    gates.add_argument("--min-gap-counts", type=float, default=1000.0)
    gates.add_argument("--pass-tol-counts", type=float, default=200.0)
    gates.add_argument("--zero-tol-counts", type=float, default=200.0)
    gates.add_argument("--span-tol", type=float, default=0.10)
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(run(args)))
    except (KvsError, CalBoardError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
