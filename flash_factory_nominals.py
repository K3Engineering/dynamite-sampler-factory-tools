"""Factory script 1 of 2: provision model-derived nominal values.

Runs after firmware flashing, before calibration. Looks up the board
model's nominal analog values and writes them to the Factory namespace
over BLE.

Usage:
    python flash_factory_nominals.py --board v700P [--address AA:BB:CC:DD:EE:FF] [--dry-run]
"""

import argparse
import asyncio
import sys

from dut_kvs import FOLDER_FACTORY, DutKvs, KvsError
from factory_data import BOARD_MODELS, nominal_entries


async def provision(args: argparse.Namespace) -> int:
    entries = nominal_entries(args.board)

    if args.dry_run:
        print(
            f"Would write to the Factory namespace ({args.address or 'auto-detected device'}):"
        )
        for key, value in entries.items():
            print(f"  {key:12s} = {value}")
        return 0

    async with await DutKvs.connect(args.address) as dut:
        failures = 0
        for key, value in entries.items():
            await dut.set(FOLDER_FACTORY, key, value)
            readback = await dut.get(FOLDER_FACTORY, key)
            ok = readback == value
            failures += not ok
            print(
                f"  {key:12s} = {value:20s} {'ok' if ok else f'MISMATCH (read {readback!r})'}"
            )

    if failures:
        print(f"FAILED: {failures} of {len(entries)} keys did not read back")
        return 1
    print(f"Factory nominals for {args.board} provisioned and verified.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--board",
        required=True,
        choices=sorted(BOARD_MODELS),
        help="board model, e.g. v700P",
    )
    parser.add_argument(
        "--address",
        help="BLE address of the DUT (default: auto-detect, only one may be in range)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be written, do not connect",
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(provision(args)))
    except KvsError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
