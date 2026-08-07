"""KVS client imports for the factory tools.

The device-neutral client lives in the sibling python-api project
(dynamite_sampler_kvs.py); this module puts it on sys.path and adds the
factory-floor alias: in calibration/provisioning flows the device is the DUT.
Dev tools (inspect_flash, edit_flash) use KvsClient directly.
"""

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

DutKvs = KvsClient

__all__ = [
    "FOLDER_FACTORY",
    "FOLDER_NAMES",
    "FOLDER_SETTINGS",
    "FOLDER_USER",
    "NVS_TYPE_STR",
    "KvsClient",
    "KvsError",
    "DutKvs",
]
