"""Inspect flash contents

Dumps the Factory, User and Settings KVS namespaces over BLE and prints
them. Pretty-prints by default; --raw emits JSON for scripting.

Never modifies the device. For targeted edits use edit_flash.py.

Usage:
    python inspect_flash.py [--address AA:BB:CC:DD:EE:FF] [--folder F [--folder U]] [--raw]
"""

import argparse
import asyncio
import json
import sys

from kvs_api_shim import (
    FOLDER_NAMES,
    AsyncDynamiteSampler,
    KvsError,
    folder_type,
    namespace,
)


async def dump_namespace(device, folder) -> dict[str, str]:
    """All key/value pairs in a namespace. String entries come from the
    snapshot; anything the firmware never writes is flagged."""
    values = device.kvs.snapshot.get(folder, {})
    keys = await namespace(device, folder).keys()
    return {key: values.get(key, "<not a string entry>") for key in keys}


def print_pretty(
    device_info: dict, data: dict[str, dict[str, str]], folders: list[str]
) -> None:
    print(
        f"Device: {device_info['address']} ({device_info['name']})"
        f"  board {device_info['board_model']}  fw {device_info['firmware_rev']}"
    )
    for folder in folders:
        ns = data[folder]
        print()
        if not ns:
            print(f"{FOLDER_NAMES[folder]} namespace (empty)")
            continue
        print(f"{FOLDER_NAMES[folder]} namespace ({len(ns)} keys)")
        width = max(len(key) for key in ns)
        for key, value in ns.items():
            print(f"  {key:{width}} = {value}")


async def inspect(args: argparse.Namespace) -> int:
    folders = args.folder or list(FOLDER_NAMES)
    async with await AsyncDynamiteSampler.connect(args.address) as device:
        device_info = {
            "address": device.info.address,
            "name": device.info.name,
            "board_model": device.info.board_model,
            "firmware_rev": device.info.firmware,
        }
        data = {folder: await dump_namespace(device, folder) for folder in folders}

    if args.raw:
        print(
            json.dumps(
                {
                    "device": device_info,
                    "namespaces": {FOLDER_NAMES[f]: data[f] for f in folders},
                },
                indent=2,
            )
        )
    else:
        print_pretty(device_info, data, folders)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        help="BLE address of the device (default: auto-detect, only one may be in range)",
    )
    parser.add_argument(
        "--folder",
        action="append",
        type=folder_type,
        help="namespace to dump: F, U or S (default: all three; repeatable)",
    )
    parser.add_argument(
        "--raw", action="store_true", help="emit JSON instead of pretty print"
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(inspect(args)))
    except KvsError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
