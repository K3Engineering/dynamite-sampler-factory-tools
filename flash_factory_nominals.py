"""Factory script 1 of 4: provision model-derived nominal values.

Runs after firmware flashing, before calibration. The board model is
auto-detected from the firmware's Hardware Revision BLE characteristic
(--board overrides), its nominal analog values are looked up and written
to the Factory namespace over BLE. To wipe the namespace instead, use
edit_flash.py.

Usage:
    python flash_factory_nominals.py [--address AA:BB:CC:DD:EE:FF] [--board v700P] [--dry-run]
"""

import argparse
import asyncio
import sys

from dut_kvs import FOLDER_FACTORY, DutKvs, HardwareRevision, KvsError
from nominal_values import BOARD_MODELS, nominal_entries
from dynamite_sampler_bleak_util import read_characteristic


async def detect_board_model(dut: DutKvs) -> str:
    """Board model compiled into the flashed firmware, e.g. 'v700P'."""
    model = await read_characteristic(dut.client, HardwareRevision)
    if not model:
        raise KvsError("Could not read the Hardware Revision characteristic")
    if model not in BOARD_MODELS:
        raise KvsError(
            f"Firmware reports unknown board model {model!r} "
            f"(known: {', '.join(sorted(BOARD_MODELS))}). Use --board to override."
        )
    return model


async def provision(args: argparse.Namespace) -> int:
    # Fully offline dry-run: no device needed.
    if args.dry_run and args.board:
        print(
            f"Would write to the Factory namespace ({args.address or 'auto-detected device'}):"
        )
        for key, value in nominal_entries(args.board).items():
            print(f"  {key:12s} = {value}")
        return 0

    async with await DutKvs.connect(args.address) as dut:
        board = args.board or await detect_board_model(dut)
        print(
            f"Board model: {board}{' (from --board)' if args.board else ' (auto-detected)'}"
        )

        if args.dry_run:
            print("Would write to the Factory namespace:")
            for key, value in nominal_entries(board).items():
                print(f"  {key:12s} = {value}")
            return 0

        failures = 0
        for key, value in nominal_entries(board).items():
            await dut.set(FOLDER_FACTORY, key, value)
            readback = await dut.get(FOLDER_FACTORY, key)
            ok = readback == value
            failures += not ok
            print(
                f"  {key:12s} = {value:20s} {'ok' if ok else f'MISMATCH (read {readback!r})'}"
            )

    if failures:
        print(f"FAILED: {failures} keys did not read back")
        return 1
    print(f"Factory nominals for {board} provisioned and verified.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--board",
        choices=sorted(BOARD_MODELS),
        help="board model (default: auto-detect from the firmware's "
        "Hardware Revision characteristic)",
    )
    parser.add_argument(
        "--address",
        help="BLE address of the DUT (default: auto-detect, only one may be in range)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be written, do not modify the device",
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(provision(args)))
    except KvsError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
