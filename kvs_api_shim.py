"""Imports for the factory tools from the sibling python-api package.

Puts python-api's ``src`` on sys.path and re-exports the names the factory
CLIs use, so they read like ordinary API users. Delete once python-api is
installed in the factory venv.

``folder_type`` and ``namespace`` are argparse/CLI plumbing that stays here.
"""

import argparse
import sys
from pathlib import Path

_PYTHON_API = Path(__file__).resolve().parent.parent / "python-api"
sys.path.insert(0, str(_PYTHON_API / "src"))

from dynamite_sampler import (  # noqa: E402
    UNCONFIGURED,
    AsyncDynamiteSampler,
    DynamiteError,
    KvsBusy,
    KvsError,
    KvsRejected,
    KvsTimeout,
)
from dynamite_sampler.kvs import (  # noqa: E402
    FOLDER_FACTORY,
    FOLDER_NAMES,
    FOLDER_SETTINGS,
    FOLDER_USER,
    KVS_WRITE_DELAY_S,
)

__all__ = [
    "UNCONFIGURED",
    "AsyncDynamiteSampler",
    "DynamiteError",
    "KvsBusy",
    "KvsError",
    "KvsRejected",
    "KvsTimeout",
    "FOLDER_FACTORY",
    "FOLDER_NAMES",
    "FOLDER_SETTINGS",
    "FOLDER_USER",
    "KVS_WRITE_DELAY_S",
    "folder_type",
    "namespace",
]


def folder_type(s: str) -> str:
    """argparse type for a KVS folder letter (case-insensitive)."""
    s = s.upper()
    if s not in FOLDER_NAMES:
        raise argparse.ArgumentTypeError(
            f"folder must be one of {', '.join(FOLDER_NAMES)} (got {s!r})"
        )
    return s


def namespace(device, folder: str):
    """The device's :class:`KvsNamespace` handle for a folder letter."""
    return {
        FOLDER_FACTORY: device.kvs.factory,
        FOLDER_USER: device.kvs.user,
        FOLDER_SETTINGS: device.kvs.settings,
    }[folder]
