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
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kvs_api_shim import FOLDER_FACTORY, KVS_WRITE_DELAY_S, KvsClient, KvsError

import dynamite_sampler_api as ds
from dynamite_sampler_bleak_util import FeedSession, read_characteristic
from capture import FeedRecorder

import cal_math
import db
from calboard_driver import CalBoard, CalBoardError
from nominal_values import BOARD_MODELS

SCRIPT_VERSION = "board_calibration 1.2"

# TODO: read the DUT's onboard temperature sensor and plumb it through
# (cal.temp dut field, segments table). Placeholder until then.
DUT_TEMP_PLACEHOLDER_C = -999.0

DEFAULT_DB_PATH = Path(__file__).resolve().with_name("factory_log.db")
DEFAULT_CAPTURE_DIR = Path(__file__).resolve().with_name("captures")

# Sweep phases in cal-board channels: one channel per bridge at a time.
# DUT channel index = cal channel - 1 (fixture wiring).
PHASES = ((1, 3), (2, 4))


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


@dataclass
class DutInfo:
    """Identity and provisioning read from the DUT before the sweep."""

    firmware_rev: str | None
    adc_config: ds.ADCConfigData  # per-channel PGA readback (span cross-check)
    board_model: str
    nominals: dict[str, float]


async def check_provisioning(dut: KvsClient, board_override: str | None) -> DutInfo:
    """Read identity + analog nominals, cross-checked against the firmware's
    hardware revision."""
    firmware_rev = await read_characteristic(dut.client, ds.DeviceInfo.FirmwareRevision)
    hw_rev = await read_characteristic(dut.client, ds.DeviceInfo.HardwareRevision)
    adc_config = await read_characteristic(dut.client, ds.DynamiteSampler.ADCConfig)
    if adc_config is None:
        raise KvsError("ADC config unreadable — cannot cross-check the span")

    board_model = board_override
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
    return DutInfo(firmware_rev, adc_config, board_model, nominals)


async def sweep(
    cal: CalBoard, recorder: FeedRecorder, dwell: float, guard: float
) -> list[dict]:
    """Step the cal board through the reversal sweep while the feed records.
    Returns per-state window metadata (SSN bounds, confirmations)."""
    seg_meta = []
    for phase_idx, (cha, chb) in enumerate(PHASES):
        for seq_idx, mv in enumerate(cal_math.SWEEP_SEQUENCE_MV):
            t_cmd = time.time()
            confirm = await asyncio.to_thread(cal.set_channels, {cha: mv, chb: mv})
            await asyncio.sleep(guard)
            ssn_start = (recorder.last_ssn or 0) + 1
            await asyncio.sleep(dwell)
            ssn_end = recorder.last_ssn
            temp_c = await asyncio.to_thread(cal.read_temperature)
            seg_meta.append(
                {
                    "phase": phase_idx,
                    "seq_idx": seq_idx,
                    "mv": mv,
                    "channels": (cha, chb),
                    "t_cmd": t_cmd,
                    "ssn_start": ssn_start,
                    "ssn_end": ssn_end,
                    "temp_c": temp_c,
                    "confirm": confirm,
                }
            )
            print(
                f"  phase {phase_idx} ch{cha}+{chb} {mv:+3d} mV "
                f"(ssn {ssn_start}..{ssn_end}, {temp_c:.2f} C)"
            )
    return seg_meta


def reduce_segments(
    recorder: FeedRecorder, seg_meta: list[dict]
) -> list[cal_math.SegmentResult]:
    """Segment the capture per commanded state (SSN window) and reduce each
    window to per-channel statistics."""
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
        segments.append(
            cal_math.SegmentResult(
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
                temp_c=meta["temp_c"],
            )
        )
    return segments


async def write_entries(dut: KvsClient, entries: dict[str, str]) -> int:
    """Write calibration entries to the Factory namespace (read-back verified).
    Returns the number of mismatched keys."""
    readbacks = await dut.set_many_verified(FOLDER_FACTORY, entries)
    mismatches = 0
    for key, value in entries.items():
        readback = readbacks[key]
        ok = readback == value
        mismatches += not ok
        print(
            f"  {key:12s} = {value:56s} "
            f"{'ok' if ok else f'MISMATCH (read {readback!r})'}"
        )
    return mismatches


async def run(args: argparse.Namespace) -> int:
    gate_params = cal_math.GateParams(
        min_window_samples=args.min_window_samples,
        max_missing=args.max_missing,
        max_std_counts=args.max_std_counts,
        min_gap_counts=args.min_gap_counts,
        pass_tol_counts=args.pass_tol_counts,
        zero_tol_counts=args.zero_tol_counts,
        span_tol=args.span_tol,
        max_temp_spread_c=args.max_temp_spread_c,
    )

    async with await KvsClient.connect(args.address) as dut:
        dut_info = await check_provisioning(dut, args.board)
        nominals = dut_info.nominals

        cal = await asyncio.to_thread(CalBoard.connect, args.cal_port)
        cal_uid = f"{await asyncio.to_thread(cal.unique_id):012X}"
        print(f"Calibration board on {cal.port}: {cal.fw_id} (uid {cal_uid})")

        # Expected +/-FS span per channel, from the flash nominals and the
        # ADC's own PGA readback — the gross-error cross-check.
        expected_span = [
            cal_math.expected_span_counts(
                nominals["adc_fsr"], nominals["afe_gain"], pga, nominals["exc"]
            )
            for pga in dut_info.adc_config.gains
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
                "device_name": dut.advertised_name,
                "board_model": dut_info.board_model,
                "firmware_rev": dut_info.firmware_rev,
                "cal_board_id": cal.fw_id,
                "cal_board_uid": cal_uid,
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
            device_info={
                "FirmwareRevision": dut_info.firmware_rev,
                "ADCConfig": dut_info.adc_config,
            },
        )
        try:
            await session.start()
            deadline = time.monotonic() + 10.0
            while recorder.last_ssn is None:
                if time.monotonic() > deadline:
                    raise KvsError("no ADC feed data within 10 s of subscribing")
                await asyncio.sleep(0.05)
            print(f"Feed running (first SSN {recorder.last_ssn}). Sweeping...")

            seg_meta = await sweep(cal, recorder, args.dwell, args.guard)

            # Stop the stream before KVS access (firmware device lock).
            await session.stop()
            await asyncio.sleep(KVS_WRITE_DELAY_S)

            segments = reduce_segments(recorder, seg_meta)
            for meta, seg in zip(seg_meta, segments):
                db.insert_segment(con, run_id, seg, confirm=meta["confirm"])

            config_values = cal_math.collect_config_values(segments)
            report = cal_math.gate_run(
                segments, config_values, expected_span, gate_params
            )
            if report.failures:
                # DB gets the headline verdicts, then the full detail list.
                headlines = [s for s in report.summary if not s.startswith(" ")]
                db.finish_run(
                    con,
                    run_id,
                    "fail",
                    "; ".join(headlines) + " | " + "; ".join(report.messages),
                )
                print(f"CALIBRATION FAILED (run #{run_id}) — nothing written")
                print()
                for line in report.summary:
                    print(f"  {line}")
                print()
                print("  detail:")
                for message in report.messages:
                    print(f"  - {message}")
                return 1

            readings = cal_math.final_readings(config_values)
            temp_calboard_mean = statistics.fmean(seg.temp_c for seg in segments)
            entries = cal_math.build_flash_entries(
                readings,
                cal.fw_id,
                exc_mv=args.exc_mv,
                now=started,
                tool=SCRIPT_VERSION,
                origin="factory",
                adc_gains=dut_info.adc_config.gains,
                temp_dut_c=DUT_TEMP_PLACEHOLDER_C,
                temp_calboard_c=temp_calboard_mean,
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

            mismatches = await write_entries(dut, entries)
            if mismatches:
                db.finish_run(con, run_id, "fail", f"{mismatches} keys misverified")
                print(f"FAILED: {mismatches} of {len(entries)} keys did not read back")
                return 1

            log_id = db.log_calibration(
                con,
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "device_address": dut.client.address,
                    "device_name": dut.advertised_name,
                    "board_model": dut_info.board_model,
                    "firmware_rev": dut_info.firmware_rev,
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
    # FeedSession.stop() relies on task cancellation reaching the feed pump;
    # the legacy asyncio.wait_for() on Python < 3.12 can silently swallow it
    # and hang the script after the sweep. Refuse to run there.
    assert sys.version_info >= (3, 12), (
        f"Python >= 3.12 required, running {sys.version.split()[0]} "
        "(3.11's asyncio.wait_for can swallow task cancellation and hang "
        "the feed shutdown)"
    )

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
    gates.add_argument("--max-std-counts", type=float, default=150.0)
    gates.add_argument("--min-gap-counts", type=float, default=1000.0)
    gates.add_argument("--pass-tol-counts", type=float, default=200.0)
    gates.add_argument("--zero-tol-counts", type=float, default=200.0)
    gates.add_argument("--span-tol", type=float, default=0.10)
    gates.add_argument("--max-temp-spread-c", type=float, default=0.2)
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(run(args)))
    except (KvsError, CalBoardError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
