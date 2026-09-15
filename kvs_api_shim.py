"""KVS imports for the factory tools — TEMPORARY shim.

The device-neutral client lives in the sibling python-api package
(``dynamite_sampler.kvs``); this module puts its ``src`` (and the flat legacy
modules) on sys.path and re-exports what the factory CLIs use. Delete once
python-api is installed in the factory venv.

folder_type is the exception: it is argparse plumbing for the factory CLIs
and stays here on purpose.
"""

import argparse
import sys
from pathlib import Path

_PYTHON_API = Path(__file__).resolve().parent.parent / "python-api"
sys.path.insert(0, str(_PYTHON_API / "src"))
sys.path.insert(0, str(_PYTHON_API))

from dynamite_sampler.kvs import (  # noqa: E402
    FOLDER_FACTORY,
    FOLDER_NAMES,
    FOLDER_SETTINGS,
    FOLDER_USER,
    KVS_WRITE_DELAY_S,
    NVS_TYPE_STR,
    KvsBusy,
    KvsClient,
    KvsDeviceError,
    KvsError,
)

__all__ = [
    "FOLDER_FACTORY",
    "FOLDER_NAMES",
    "FOLDER_SETTINGS",
    "FOLDER_USER",
    "KVS_WRITE_DELAY_S",
    "NVS_TYPE_STR",
    "KvsBusy",
    "KvsClient",
    "KvsDeviceError",
    "KvsError",
    "folder_type",
]


def folder_type(s: str) -> str:
    """argparse type for a KVS folder letter (case-insensitive)."""
    s = s.upper()
    if s not in FOLDER_NAMES:
        raise argparse.ArgumentTypeError(
            f"folder must be one of {', '.join(FOLDER_NAMES)} (got {s!r})"
        )
    return s
