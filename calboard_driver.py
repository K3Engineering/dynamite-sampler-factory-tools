"""Driver for the MicroPython calibration board over the raw REPL (mpremote).

The board (ESP32-S3 running firmware/calibration-board-fw) exposes no command
protocol — main.py leaves a global `board` object in the REPL namespace and
this driver wraps it in plain Python calls over USB serial.

    with CalBoard.connect() as cal:        # or CalBoard.connect("COM5")
        cal.set_channels({1: 5, 3: -5})
        cal.zero_bridges()

Port auto-detection probes plausible serial ports (ESP32/bridge VIDs only)
for a MicroPython REPL that can construct a CalibrationBoard; pass the port
explicitly to skip probing.
"""

import serial.tools.list_ports
from mpremote.transport import TransportError, TransportExecError
from mpremote.transport_serial import SerialTransport

# USB VIDs the calibration board may enumerate as: ESP32-S3 native USB or the
# usual UART bridge chips. Probing enters the raw REPL (sends control chars),
# so ports on other devices are never touched — pass the port explicitly if
# the board is behind something else.
_PROBE_VIDS = {
    0x303A,  # Espressif native USB
    0x10C4,  # CP210x
    0x1A86,  # CH340/CH9102
    0x0403,  # FTDI
}

# Ensures the `board` global exists even if main.py didn't run (e.g. after a
# manual soft reboot or a crashed session).
_BOOTSTRAP = """
try:
    board
except NameError:
    from calboard import CalibrationBoard
    board = CalibrationBoard()
"""


def _candidate_ports() -> list[str]:
    return [p.device for p in serial.tools.list_ports.comports() if p.vid in _PROBE_VIDS]


class CalBoardError(Exception):
    """The board is unreachable, is not a calibration board, or rejected a command."""


class CalBoard:
    """An open raw-REPL session to the calibration board."""

    def __init__(self, transport: SerialTransport, port: str, fw_id: str):
        self._transport = transport
        self.port = port
        self.fw_id = fw_id  # calboard.py CalibrationBoard.FW_ID ("unknown" on old fw)

    @classmethod
    def connect(cls, port: str | None = None) -> "CalBoard":
        """Open the calibration board; probe plausible ports unless told."""
        candidates = [port] if port else _candidate_ports()
        if not candidates:
            raise CalBoardError(
                "no candidate serial ports (probed VIDs: "
                f"{', '.join(f'{vid:#06x}' for vid in sorted(_PROBE_VIDS))}); "
                "pass --cal-port"
            )
        errors = []
        for candidate in candidates:
            transport = None
            try:
                transport = SerialTransport(candidate, 115200)
                transport.enter_raw_repl(soft_reset=False, timeout_overall=3)
                transport.exec(_BOOTSTRAP)
                fw_id = transport.eval("getattr(board, 'FW_ID', 'unknown')")
                return cls(transport, candidate, str(fw_id))
            except Exception as e:  # not our board / not a REPL: keep looking
                errors.append(f"  {candidate}: {e}")
                if transport is not None:
                    try:
                        transport.close()
                    except Exception:
                        pass
        raise CalBoardError(
            "no calibration board found on any serial port:\n" + "\n".join(errors)
        )

    def _exec(self, code: str) -> str:
        """Run a statement on the board; returns its stdout (confirmations)."""
        try:
            return self._transport.exec(code).decode().strip()
        except TransportExecError as e:
            raise CalBoardError(f"board rejected {code!r}: {e}") from e
        except TransportError as e:
            raise CalBoardError(f"transport error during {code!r}: {e}") from e

    def _eval(self, expr: str):
        """Evaluate an expression on the board; returns the parsed value."""
        try:
            return self._transport.eval(expr)
        except TransportExecError as e:
            raise CalBoardError(f"board rejected {expr!r}: {e}") from e
        except TransportError as e:
            raise CalBoardError(f"transport error during {expr!r}: {e}") from e

    def read_temperature(self) -> float:
        """TMP118 temperature of the calibration board, °C."""
        return float(self._eval("therm.celsius()"))

    def unique_id(self) -> int:
        """TMP118 48-bit unique ID — the physical board's identity."""
        return int(self._eval("therm.unique_id()"))

    def set_channels(self, channel_voltages: dict) -> str:
        """{channel: mV}; at most one channel per bridge (calboard.py rule)."""
        if not channel_voltages:
            return ""
        return self._exec(f"board.set_channels({channel_voltages!r})")

    def zero_bridges(self) -> str:
        return self._exec("board.zero_bridges()")

    def show_state(self) -> str:
        return self._exec("board.show_state()")

    def close(self) -> None:
        try:
            self._transport.exit_raw_repl()
        except Exception:
            pass
        self._transport.close()

    def __enter__(self) -> "CalBoard":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
