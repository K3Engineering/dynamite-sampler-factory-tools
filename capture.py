"""Block-consuming capture sinks for the factory tools.

Raw ADC feed to CSV: ``ssn, t_unix_ms, ch0..ch3``, one row per received
sample. Dropped samples arrive as all-NaN rows in a Block and are counted,
not written — the SSN column skips them, matching the on-disk format the
cal-board and Allan captures have always used.
"""

import asyncio
import csv
import pathlib
import time
from datetime import datetime, timezone

import numpy as np

_CAPTURE_COLUMNS = ("ssn", "t_unix_ms", "ch0", "ch1", "ch2", "ch3")


class CsvCapture:
    """Writes a raw feed to CSV; counts dropped (NaN) rows and tracks SSNs."""

    COLUMNS = _CAPTURE_COLUMNS

    def __init__(self, file_path, device_dict):
        self.file_path = pathlib.Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.file_path, "w", newline="")
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"# captured: {stamp}", file=self._file)
        print(f"# device: {device_dict}", file=self._file)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._last_ssn = None
        self._missing = 0

    @property
    def last_ssn(self):
        """Highest unwrapped SSN seen, including dropped ones; None before
        the first block."""
        return self._last_ssn

    @property
    def missing_count(self):
        return self._missing

    def add_block(self, block):
        """Write a raw-unit Block's received rows; returns them as
        ``(ssn, t_unix_ms, ch0..ch3)`` tuples."""
        valid = ~np.isnan(block.raw).any(axis=1)
        self._missing += int((~valid).sum())
        self._last_ssn = block.ssn0 + block.raw.shape[0] - 1
        rows = []
        if valid.any():
            t_ms = round(time.time() * 1000)
            for i in np.nonzero(valid)[0]:
                row = (block.ssn0 + int(i), t_ms, *(int(v) for v in block.raw[i]))
                self._writer.writerow(row)
                rows.append(row)
        self._file.flush()
        return rows

    def close(self):
        self._file.close()


class FeedPump:
    """Pumps a raw device stream into a recorder in a background task, so the
    caller can operate a stimulus (relay sweep, timed capture) meanwhile.

    A pump failure (``ConnectionLost``, ``BufferOverrun``) is re-raised by
    :meth:`check` from the control loop and by :meth:`stop`; a run must not
    finish with a dead pump."""

    def __init__(self, device, recorder, blocksize=100):
        self._task = asyncio.ensure_future(self._run(device, recorder, blocksize))

    @staticmethod
    async def _run(device, recorder, blocksize):
        async for block in device.stream(blocksize=blocksize, units="raw"):
            recorder.add_block(block)

    def check(self):
        """Re-raise a pump failure, if any."""
        if self._task.done() and not self._task.cancelled():
            self._task.result()

    async def stop(self):
        """Stop the pump; a failure that already ended it is re-raised."""
        self.check()
        await self._cancel()

    async def cancel(self):
        """Stop the pump, swallowing any failure (cleanup paths)."""
        await self._cancel()

    async def _cancel(self):
        if self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
