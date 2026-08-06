"""Factory script 3 of 4: inspect flash contents (read-only).

Dumps the Factory, User and Settings KVS namespaces over BLE and prints
them. Pretty-prints by default; --raw emits JSON for scripting. Never
modifies the device. For targeted edits use edit_flash.py.

Usage:
    python inspect_flash.py [--address AA:BB:CC:DD:EE:FF] [--folder F [--folder U]] [--raw]
"""

import argparse
import asyncio
import json
import sys

from dut_kvs import (
    FOLDER_NAMES,
    NVS_TYPE_NAMES,
    NVS_TYPE_STR,
    DutKvs,
    HardwareRevision,
    KvsError,
)
import dynamite_sampler_api as ds
from dynamite_sampler_bleak_util import read_characteristic


def folder_type(s: str) -> str:
    s = s.upper()
    if s not in FOLDER_NAMES:
        raise argparse.ArgumentTypeError(
            f"folder must be one of {', '.join(FOLDER_NAMES)} (got {s!r})"
        )
    return s


async def dump_namespace(dut: DutKvs, folder: str) -> dict[str, str]:
    """All key/value pairs in a namespace. Only strings are readable over
    the KVS protocol; other types are shown as placeholders."""
    out = {}
    for key, nvs_type in await dut.list_entries(folder):
        if nvs_type == NVS_TYPE_STR:
            try:
                out[key] = await dut.get(folder, key)
            except KvsError:
                out[key] = "<unreadable>"
        else:
            type_name = NVS_TYPE_NAMES.get(nvs_type, hex(nvs_type))
            out[key] = f"<{type_name}, not readable over KVS>"
    return out


def print_pretty(device: dict, data: dict[str, dict[str, str]], folders: list[str]) -> None:
    print(
        f"Device: {device['address']} ({device['name']})"
        f"  board {device['board_model']}  fw {device['firmware_rev']}"
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
    async with await DutKvs.connect(args.address) as dut:
        device = {
            "address": dut.client.address,
            "name": dut.device_name,
            "board_model": await read_characteristic(dut.client, HardwareRevision),
            "firmware_rev": await read_characteristic(dut.client, ds.DeviceInfo.FirmwareRevision),
        }
        data = {folder: await dump_namespace(dut, folder) for folder in folders}

    if args.raw:
        print(
            json.dumps(
                {
                    "device": device,
                    "namespaces": {FOLDER_NAMES[f]: data[f] for f in folders},
                },
                indent=2,
            )
        )
    else:
        print_pretty(device, data, folders)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        help="BLE address of the DUT (default: auto-detect, only one may be in range)",
    )
    parser.add_argument(
        "--folder",
        action="append",
        type=folder_type,
        help="namespace to dump: F, U or S (default: all three; repeatable)",
    )
    parser.add_argument("--raw", action="store_true", help="emit JSON instead of pretty print")
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(inspect(args)))
    except KvsError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
