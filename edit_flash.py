"""Factory script 4 of 4: targeted edits to flash contents (dev tool).

Get/set/delete individual keys, or clear a whole namespace, in any of the
KVS folders (Factory, User, Settings). For read-only dumps use
inspect_flash.py.

Usage:
    python edit_flash.py get F exc
    python edit_flash.py set F exc 4.53,nominal
    python edit_flash.py del F exc
    python edit_flash.py clear F [--yes]
"""

import argparse
import asyncio
import sys

from dut_kvs import FOLDER_NAMES, DutKvs, KvsError


def folder_type(s: str) -> str:
    s = s.upper()
    if s not in FOLDER_NAMES:
        raise argparse.ArgumentTypeError(
            f"folder must be one of {', '.join(FOLDER_NAMES)} (got {s!r})"
        )
    return s


async def cmd_get(dut: DutKvs, args: argparse.Namespace) -> int:
    print(await dut.get(args.folder, args.key))
    return 0


async def cmd_set(dut: DutKvs, args: argparse.Namespace) -> int:
    await dut.set(args.folder, args.key, args.value)
    readback = await dut.get(args.folder, args.key)
    ok = readback == args.value
    print(f"{args.folder}.{args.key} = {readback} {'ok' if ok else 'MISMATCH'}")
    return 0 if ok else 1


async def cmd_del(dut: DutKvs, args: argparse.Namespace) -> int:
    await dut.delete(args.folder, args.key)
    print(f"Deleted {args.folder}.{args.key}")
    return 0


async def cmd_clear(dut: DutKvs, args: argparse.Namespace) -> int:
    """Delete every key in the namespace."""
    keys = await dut.keys(args.folder)
    if not keys:
        print(f"{FOLDER_NAMES[args.folder]} namespace is already empty.")
        return 0

    print(f"{len(keys)} keys in the {FOLDER_NAMES[args.folder]} namespace:")
    for key in keys:
        try:
            print(f"  {key:12s} = {await dut.get(args.folder, key)}")
        except KvsError:
            print(f"  {key:12s}   (unreadable)")

    if not args.yes:
        reply = input(f"Delete all {len(keys)} keys? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 2

    for key in keys:
        await dut.delete(args.folder, key)

    remaining = await dut.keys(args.folder)
    if remaining:
        print(f"FAILED: {len(remaining)} keys remain: {', '.join(remaining)}")
        return 1
    print(f"{FOLDER_NAMES[args.folder]} namespace cleared.")
    return 0


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--address",
        help="BLE address of the DUT (default: auto-detect, only one may be in range)",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("get", parents=[common], help="read a key")
    p.add_argument("folder", type=folder_type)
    p.add_argument("key")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser(
        "set", parents=[common], help="write a key, with readback verify"
    )
    p.add_argument("folder", type=folder_type)
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("del", parents=[common], help="delete a key")
    p.add_argument("folder", type=folder_type)
    p.add_argument("key")
    p.set_defaults(func=cmd_del)

    p = sub.add_parser("clear", parents=[common], help="delete all keys in a namespace")
    p.add_argument("folder", type=folder_type)
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    async def run() -> int:
        async with await DutKvs.connect(args.address) as dut:
            return await args.func(dut, args)

    try:
        sys.exit(asyncio.run(run()))
    except (KvsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
