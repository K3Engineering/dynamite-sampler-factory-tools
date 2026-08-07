"""KVS imports for the factory tools — TEMPORARY shim.

The device-neutral client lives in the sibling python-api project
(dynamite_sampler_kvs.py); this module puts it on sys.path and re-exports it.
Once python-api becomes an installable package, delete this shim and import
from dynamite_sampler_kvs directly.

folder_type is the exception: it is argparse plumbing for the factory CLIs
and stays here on purpose.
"""

import argparse
import sys
from pathlib import Path

# The python-api folder is a sibling of this project and uses flat imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python-api"))

from dynamite_sampler_kvs import (  # noqa: E402
    FOLDER_FACTORY,
    FOLDER_NAMES,
    FOLDER_SETTINGS,
    FOLDER_USER,
    KVS_WRITE_DELAY_S,
    NVS_TYPE_STR,
    KvsClient,
    KvsError,
)

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
]


def folder_type(s: str) -> str:
    """argparse type for a KVS folder letter (case-insensitive)."""
    s = s.upper()
    if s not in FOLDER_NAMES:
        raise argparse.ArgumentTypeError(
            f"folder must be one of {', '.join(FOLDER_NAMES)} (got {s!r})"
        )
    return s
