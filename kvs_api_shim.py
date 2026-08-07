"""Temporary KVS client imports shim for the factory tools.

Also a temporary place for things that should move to their own file or into the API

The device-neutral client lives in the sibling python-api project
(dynamite_sampler_kvs.py); this module puts it on sys.path and adds the
factory-floor alias: in calibration/provisioning flows the device is the DUT.
Dev tools (inspect_flash, edit_flash) use KvsClient directly.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# The python-api folder is a sibling of this project and uses flat imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python-api"))

from dynamite_sampler_kvs import (  # noqa: E402
    FOLDER_FACTORY,
    FOLDER_NAMES,
    FOLDER_SETTINGS,
    FOLDER_USER,
    NVS_TYPE_STR,
    KvsClient,
    KvsError,
)

# Grace between retried writes: KVS commands are rejected while the device
# is busy (firmware device lock), and state changes take a moment to settle.
KVS_WRITE_DELAY_S = 0.5

__all__ = [
    "FOLDER_FACTORY",
    "FOLDER_NAMES",
    "FOLDER_SETTINGS",
    "FOLDER_USER",
    "KVS_WRITE_DELAY_S",
    "NVS_TYPE_STR",
    "KvsClient",
    "KvsError",
    "folder_type",
    "set_many_verified",
    "set_verified",
]


def folder_type(s: str) -> str:
    """argparse type for a KVS folder letter (case-insensitive)."""
    s = s.upper()
    if s not in FOLDER_NAMES:
        raise argparse.ArgumentTypeError(
            f"folder must be one of {', '.join(FOLDER_NAMES)} (got {s!r})"
        )
    return s


async def set_verified(
    client: KvsClient, folder: str, key: str, value: str, attempts: int = 3
) -> str:
    """SET + read-back verify, with retries (device-lock grace). Returns the
    readback (compare against `value`); re-raises KvsError after retries."""
    for attempt in range(attempts):
        try:
            await client.set(folder, key, value)
            return await client.get(folder, key)
        except KvsError:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(KVS_WRITE_DELAY_S)
    raise ValueError(f"attempts must be >= 1 (got {attempts})")


async def set_many_verified(
    client: KvsClient, folder: str, entries: dict[str, str]
) -> dict[str, str]:
    """set_verified over a {key: value} mapping; returns {key: readback}."""
    return {
        key: await set_verified(client, folder, key, value)
        for key, value in entries.items()
    }
