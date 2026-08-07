"""Flash nominal values (EXC, etc)

Run after firmware flashing, before calibration. The board model is
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

from kvs_api_shim import FOLDER_FACTORY, KvsClient, KvsError
from nominal_values import BOARD_MODELS, nominal_entries
import dynamite_sampler_api as ds
from dynamite_sampler_bleak_util import read_characteristic


async def detect_board_model(device: KvsClient) -> str:
    """Board model compiled into the flashed firmware, e.g. 'v700P'."""
    model = await read_characteristic(device.client, ds.DeviceInfo.HardwareRevision)
    if not model:
        raise KvsError("Could not read the Hardware Revision characteristic")
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

    async with await KvsClient.connect(args.address) as device:
        detected = await detect_board_model(device)
        board = args.board or detected
        if args.board and args.board != detected:
            raise KvsError(
                f"--board {args.board} does not match the firmware's "
                f"Hardware Revision ({detected!r})"
            )
        if board not in BOARD_MODELS:
            raise KvsError(
                f"Unknown board model {board!r} "
                f"(known: {', '.join(sorted(BOARD_MODELS))}). "
                "Update BOARD_MODELS in nominal_values.py."
            )

        print(
            f"Board model: {board}"
            f"{' (from --board, matches firmware)' if args.board else ' (auto-detected)'}"
        )

        if args.dry_run:
            print("Would write to the Factory namespace:")
            for key, value in nominal_entries(board).items():
                print(f"  {key:12s} = {value}")
            return 0

        failures = 0
        for key, value in nominal_entries(board).items():
            readback = await device.set_verified(FOLDER_FACTORY, key, value)
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
        help="BLE address of the device (default: auto-detect, only one may be in range)",
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
