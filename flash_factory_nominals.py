"""Flash nominal values (EXC, etc)

Run after firmware flashing, before calibration. With unified firmware,
--board is mandatory, and `board_model` is written to the Factory namespace.

If the device already carries a different identity, the script refuses to
proceed; erase flash to re-provision. To wipe the Factory namespace instead,
use edit_flash.py.

Usage:
    python flash_factory_nominals.py --board v700P [--address AA:BB:CC:DD:EE:FF] [--dry-run]
"""

import argparse
import asyncio
import sys

from kvs_api_shim import UNCONFIGURED, AsyncDynamiteSampler, KvsError
from nominal_values import BOARD_MODELS, nominal_entries


async def provision(args: argparse.Namespace) -> int:
    # Fully offline dry-run: no device needed.
    if args.dry_run and not args.address:
        print("Would write to the Factory namespace (auto-detected device):")
        for key, value in nominal_entries(args.board).items():
            print(f"  {key:12s} = {value}")
        return 0

    async with await AsyncDynamiteSampler.connect(args.address) as device:
        # DIS Hardware Revision: the flashed identity, or UNCONFIGURED in
        # safe mode.
        reported = device.info.board_model
        if reported not in (UNCONFIGURED, args.board):
            raise KvsError(
                f"Device already has a different identity ({reported!r}); "
                "erase flash to re-provision."
            )
        needs_reboot = reported == UNCONFIGURED
        print(
            f"Board model: {args.board} "
            f"({'unconfigured device, writing identity' if needs_reboot else 'matches device identity'})"
        )

        if args.dry_run:
            print("Would write to the Factory namespace:")
            for key, value in nominal_entries(args.board).items():
                print(f"  {key:12s} = {value}")
            return 0

        failures = 0
        for key, value in nominal_entries(args.board).items():
            try:
                readback = await device.kvs.factory.set(key, value)
            except KvsError as e:
                failures += 1
                print(f"  {key:12s} = {value:20s} MISMATCH ({e})")
                continue
            print(f"  {key:12s} = {readback:20s} ok")

    if failures:
        print(f"FAILED: {failures} keys did not read back")
        return 1
    print(f"Factory nominals for {args.board} provisioned and verified.")
    if needs_reboot:
        print("Reboot the device for the new identity to take effect.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--board",
        required=True,
        choices=sorted(BOARD_MODELS),
        help="board model to write as the device identity",
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
