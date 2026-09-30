"""Feed pumping for the factory tools.

Captures are dynamite-csv files written by the library's ``CsvRecorder``
(one shared format with the app and python-api; dropped samples are blank
rows in the file). What stays here: running a device stream in a background
task while the caller operates a stimulus (relay sweep, timed capture).
"""

import asyncio


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
