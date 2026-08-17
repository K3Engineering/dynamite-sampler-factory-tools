"""Stream ADC noise from the DUT and plot overlapping Allan deviation.

Flow:
  1. Connect to the DUT over BLE; read the provisioning (board model,
     nominals) and the ADC config (sample rate, per-channel PGA).
     Optionally, connect to the calibration board and set up shorts
  2. Record a contiguous capture to CSV. A single dropped sample aborts the
     run — Allan math requires a gap-free record.
     3. Per channel: overlapping Allan deviation on an 8-points/octave tau grid,
     converted to nV referred to the AFE input, plus a Hann amplitude
     spectrum and a Welch amplitude spectral density. Narrowband peaks
     (Hann FFT, peaks > 6x the median bin) are listed in the report and
     marked on the spectrum. Three PNGs (ADEV, per-channel spectrum, per-channel PSD)
     are saved next to the CSV, a .txt copy of the console output (minus
     the live progress line) alongside; windows are shown unless --no-show.

Modes:
  passive (default): measure whatever is attached (load cells, bench
      shorts). --channels selects DUT channels (default: all four).
  --use-calboard: the calibration board dead-shorts the selected inputs
      (relay 0 position), giving the analog chain's noise floor. At most one
      channel per bridge, so shorting all four takes two parts (board
      channels 1+3, then 2+4) and roughly twice --duration of wall time;
      --duration applies per part. --calboard-channels (DUT numbering)
      restricts the set to a single part. The board's TMP118 temperature is
      read at the end of each part.

Common commands:
  # LCs attached, all channels, default 60 s:
  python allan_plot.py

  # cal board shorting DUT ch0+ch2 (board ch1+ch3), 30 minutes:
  python allan_plot.py --use-calboard --calboard-channels 0,2 --duration 1800

  # one load cell on ch0, 24 hours:
  python allan_plot.py --channels 0 --duration 86400

The tool only reads the device: nothing is written to flash, KVS, or the
factory DB.

TODO: read the DUT's onboard temperature sensor once exposed over BLE (same
placeholder situation as DUT_TEMP_PLACEHOLDER_C in board_calibration.py) and
annotate the plot with it — the drift leg of the Allan plot is temperature.
"""

import argparse
import asyncio
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kvs_api_shim import KvsClient, KvsError

import dynamite_sampler_api as ds
from dynamite_sampler_bleak_util import (
    FeedSession,
    NotifyCallbackFeeddatas,
    read_characteristic,
    write_characteristic,
)
from capture import FeedRecorder

import allan_math
from board_calibration import PHASES, check_provisioning
from calboard_driver import CalBoard, CalBoardError
from nominal_values import BOARD_MODELS

SCRIPT_VERSION = "allan_plot 1.3.0"

DEFAULT_CAPTURE_DIR = Path(__file__).resolve().with_name("captures")

# Wait after shorting before the window opens. The open-input high-gain
# chain was observed to take up to ~1 s to settle; 2 s guards with 2x margin.
DEFAULT_GUARD_S = 2.0

# Buffer slack beyond exact accounting, for the feed-startup latency before
# window 1 (BLE subscribe + first relay command).
CAPACITY_SLACK_S = 10.0


def estimate_capacity(rate, duration, n_parts, guard):
    """Buffer size for a capture: every streamed sample is buffered, including
    guard and startup rows outside the analysis windows."""
    return int(rate * (duration * n_parts + guard * n_parts + CAPACITY_SLACK_S)) + 64


class AllanError(Exception):
    """The capture or the channel plan is unusable."""


class StdoutTee:
    """stdout duplicator for the .txt report; drops the \r live-progress writes."""

    def __init__(self, stream, path):
        self.stream = stream
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        if not s.startswith("\r"):
            self._file.write(s)

    def flush(self):
        self.stream.flush()
        self._file.flush()

    def isatty(self):
        return self.stream.isatty()

    def close(self):
        self._file.close()


class AllanRecorder(NotifyCallbackFeeddatas):
    """NotifyCallbackFeeddatas sink: entire feed to CSV, requested channels to
    a preallocated buffer.

    Unlike FeedRecorder (list of rows), buffers are per-channel int32 arrays
    sized from the expected sample count — a 24-hour capture must not hold
    Python rows in memory. Gaps are counted, not fatal here; the capture loop
    aborts on them.
    """

    COLUMNS = FeedRecorder.COLUMNS  # same CSV format as the cal captures

    def __init__(self, file_path, keep_channels, capacity):
        self.file_path = Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.file_path, "w", newline="")
        self._writer = None
        self._keep = tuple(keep_channels)
        self._buf = {ch: np.empty(capacity, dtype=np.int32) for ch in self._keep}
        self._capacity = capacity
        self._n = 0
        self._missing = 0
        self.overflowed = False

    def setup(self, device_dict):
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"# captured: {stamp}", file=self._file)
        print(f"# device: {device_dict}", file=self._file)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def callback(self, header, feeddatas, missing):
        if feeddatas:
            t_ms = round(time.time() * 1000)
            base = header.sample_sequence_number  # already unwrapped upstream
            for i, d in enumerate(feeddatas):
                self._writer.writerow((base + i, t_ms, d.ch0, d.ch1, d.ch2, d.ch3))
            end = self._n + len(feeddatas)
            if end > self._capacity:
                self.overflowed = True
            else:
                for ch in self._keep:
                    self._buf[ch][self._n : end] = [
                        getattr(d, f"ch{ch}") for d in feeddatas
                    ]
                self._n = end
        self._missing += missing
        self._file.flush()

    @property
    def n_samples(self):
        return self._n

    @property
    def missing_count(self):
        return self._missing

    def window(self, ch, i0, i1):
        """A copy of channel ch samples [i0, i1)."""
        return self._buf[ch][i0:i1].copy()

    def cleanup(self):
        self._file.close()


def channel_list_arg(s: str) -> tuple:
    """argparse type: comma list of DUT channel numbers, e.g. "0,2"."""
    try:
        channels = tuple(sorted({int(p) for p in s.split(",")}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad channel list {s!r} (want e.g. '0,2')")
    if not channels or min(channels) < 0 or max(channels) > 3:
        raise argparse.ArgumentTypeError(f"channels must be in 0..3 (got {s!r})")
    return channels


def plan_capture(args) -> tuple[list | None, tuple]:
    """(channel sets in board channels, kept DUT channels) from the flags.

    channel sets is None in passive mode. Board channel = DUT channel + 1
    (fixture wiring, as in board_calibration).
    """
    if args.calboard_channels is not None and not args.use_calboard:
        raise AllanError("--calboard-channels requires --use-calboard")
    if args.use_calboard and args.channels is not None:
        raise AllanError(
            "--use-calboard picks the plotted channels; --channels is passive-only"
        )
    if not args.use_calboard:
        return None, args.channels or (0, 1, 2, 3)
    if args.calboard_channels is None:
        return list(PHASES), (0, 1, 2, 3)
    board = tuple(c + 1 for c in args.calboard_channels)
    # Bridges are {1,2} and {3,4}: at most one channel per bridge at a time.
    if sum(c <= 2 for c in board) > 1 or sum(c >= 3 for c in board) > 1:
        raise AllanError(
            f"calboard set {board} puts two channels on one bridge "
            "(bridges are 1/2 and 3/4)"
        )
    return [board], args.calboard_channels


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


async def capture_for(recorder: AllanRecorder, seconds: float, label: str) -> None:
    """Stream for the given time; bail out the moment the record is gapped.

    On a terminal, shows a live one-line progress report; when stdout is a
    file/pipe, the capture start/done prints are the log instead.
    """
    interactive = sys.stdout.isatty()
    start = recorder.n_samples
    deadline = time.monotonic() + seconds
    while (remaining := deadline - time.monotonic()) > 0:
        if recorder.overflowed or recorder.missing_count:
            if interactive:
                print()  # don't leave the error on the progress line
            if recorder.overflowed:
                raise AllanError(
                    "captured more samples than provisioned — capture untrusted"
                )
            raise AllanError(
                f"BLE dropped {recorder.missing_count} samples — "
                "Allan math needs a gap-free record"
            )
        if interactive:
            n = recorder.n_samples - start
            print(
                f"\r{label}: {_hms(seconds - remaining)}/{_hms(seconds)} — "
                f"{n:,} samples  ",
                end="",
                flush=True,
            )
        await asyncio.sleep(min(1.0, remaining))
    if interactive:
        print()


SPEC_PLOT_BINS = 16384


def _peak_hold_downsample(freqs, spec, n_out=SPEC_PLOT_BINS):
    n = freqs.size
    if n <= n_out:
        return freqs, spec
    edges = np.linspace(0, n, n_out + 1).astype(int)
    idx = np.array(
        [edges[i] + int(np.argmax(spec[edges[i] : edges[i + 1]])) for i in range(n_out)]
    )
    return freqs[idx], spec[idx]


@dataclass
class Trace:
    """One plotted channel: ADEV, spectrum, ASD — all nV referred to AFE input."""

    dut_ch: int
    taus_s: np.ndarray
    adev_nv: np.ndarray
    err_nv: np.ndarray
    peaks: list  # (freq_hz, amplitude_counts) from allan_math.pick_peaks
    nv_per_count: float
    spec_hz: np.ndarray
    spec_nv: np.ndarray
    asd_hz: np.ndarray
    asd_nv: np.ndarray


def reduce_traces(recorder, windows, rate, volts_per_count_ch) -> list[Trace]:
    """Cut the capture into per-channel series and run the Allan reduction."""
    traces = []
    for w in windows:
        for dut_ch in w["dut_channels"]:
            samples = recorder.window(dut_ch, w["i0"], w["i1"])
            if abs(samples.mean()) >= (
                allan_math.RAIL_FRACTION * allan_math.ADC_COUNTS_PER_POLARITY
            ):
                print(f"  ch{dut_ch}: pinned at full scale — excluded (unplugged?)")
                continue
            taus, adev, err = allan_math.overlapping_adev(samples, rate)
            spec_hz, spec = allan_math.amplitude_spectrum(samples, rate)
            peaks = allan_math.pick_peaks(spec_hz, spec)
            spec_hz, spec = _peak_hold_downsample(spec_hz, spec)
            asd_hz, asd = allan_math.noise_asd(samples, rate)
            nv_per_count = volts_per_count_ch[dut_ch] * 1e9
            traces.append(
                Trace(
                    dut_ch,
                    taus,
                    adev * nv_per_count,
                    err * nv_per_count,
                    peaks,
                    nv_per_count,
                    spec_hz,
                    spec * nv_per_count,
                    asd_hz,
                    asd * nv_per_count,
                )
            )
    return traces


def _annotate_footer(fig, meta_lines):
    if len(meta_lines) > 1:
        fig.text(0.01, 0.012, "\n".join(meta_lines[1:]), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))


def _plot_adev(plt, traces, meta_lines):
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, t in enumerate(traces):
        ax.errorbar(
            t.taus_s,
            t.adev_nv,
            yerr=t.err_nv,
            marker=".",
            markersize=3,
            linewidth=1,
            elinewidth=0.7,
            capsize=2,
            color=f"C{i}",
            label=f"ch{t.dut_ch}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("averaging time τ (s)")
    ax.set_ylabel("Allan deviation (nV) — referred to AFE input")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend()
    ax.set_title(meta_lines[0], fontsize=10)
    _annotate_footer(fig, meta_lines)
    return fig


def _plot_spectrum(plt, traces, meta_lines):
    n = len(traces)
    fig, axes = plt.subplots(
        n, 1, figsize=(9, 2.2 * n + 1.4), sharex=True, squeeze=False
    )
    for i, (ax, t) in enumerate(zip(axes[:, 0], traces)):
        color = f"C{i}"
        ax.semilogy(t.spec_hz, np.maximum(t.spec_nv, 1e-30), color=color, lw=0.7)
        if t.peaks:
            pf = [f for f, _ in t.peaks]
            pa = [max(a * t.nv_per_count, 1e-30) for f, a in t.peaks]
            ax.plot(pf, pa, "o", ms=4, color=color)
        ax.set_ylabel(f"ch{t.dut_ch} (nV)")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    axes[-1, 0].set_xlabel("frequency (Hz)")
    fig.suptitle(
        meta_lines[0].replace("Allan deviation", "Amplitude spectrum", 1),
        fontsize=10,
    )
    _annotate_footer(fig, meta_lines)
    return fig


def _plot_psd(plt, traces, meta_lines):
    n = len(traces)
    fig, axes = plt.subplots(
        n, 1, figsize=(9, 2.2 * n + 1.4), sharex=True, squeeze=False
    )
    for i, (ax, t) in enumerate(zip(axes[:, 0], traces)):
        mask = t.asd_hz > 0
        ax.loglog(
            t.asd_hz[mask],
            np.maximum(t.asd_nv[mask], 1e-30),
            color=f"C{i}",
            lw=0.9,
        )
        ax.set_ylabel(f"ch{t.dut_ch}\n(nV/√Hz)")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    axes[-1, 0].set_xlabel("frequency (Hz)")
    fig.suptitle(
        meta_lines[0].replace("Allan deviation", "Noise density", 1),
        fontsize=10,
    )
    _annotate_footer(fig, meta_lines)
    return fig


def make_plots(traces, meta_lines, png_path, no_show):
    import matplotlib

    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec_path = png_path.with_name(png_path.stem + "_spectrum.png")
    psd_path = png_path.with_name(png_path.stem + "_psd.png")
    _plot_adev(plt, traces, meta_lines).savefig(png_path, dpi=150)
    _plot_spectrum(plt, traces, meta_lines).savefig(spec_path, dpi=150)
    _plot_psd(plt, traces, meta_lines).savefig(psd_path, dpi=150)
    if not no_show:
        plt.show()
    plt.close("all")
    return png_path, spec_path, psd_path


async def run(args: argparse.Namespace) -> int:
    channel_sets, kept_channels = plan_capture(args)

    async with await KvsClient.connect(args.address) as dut:
        info = await check_provisioning(dut, args.board)
        if args.tx_power is not None:
            print(f"Setting TX power to {args.tx_power} dBm")
            await write_characteristic(dut.client, ds.TxPower.TxPowerSet, args.tx_power)
        # Always recorded — it's a measurement condition. Read back after the
        # write too: the firmware only logs a failed setPower, nothing comes
        # back over BLE.
        tx_power_dbm = await read_characteristic(dut.client, ds.DeviceInfo.TxPowerLevel)
        if args.tx_power is not None and tx_power_dbm != args.tx_power:
            raise AllanError(
                f"TX power readback {tx_power_dbm} dBm != requested {args.tx_power} dBm"
            )
        rate = info.adc_config.sample_rate
        volts_per_count_ch = [
            allan_math.volts_per_count(
                info.nominals["adc_fsr"], info.nominals["afe_gain"], pga
            )
            for pga in info.adc_config.gains
        ]

        cal = None
        cal_uid = None
        if args.use_calboard:
            cal = await asyncio.to_thread(CalBoard.connect, args.cal_port)
            cal_uid = f"{await asyncio.to_thread(cal.unique_id):012X}"
            print(f"Calibration board on {cal.port}: {cal.fw_id} (uid {cal_uid})")

        started = datetime.now(timezone.utc)
        safe_mac = "".join(c for c in dut.client.address if c.isalnum())
        csv_path = (
            args.capture_dir
            / started.strftime("%Y%m%d")
            / f"allan_{started:%Y%m%d_%H%M%S}_{safe_mac}.csv"
        )
        png_path = csv_path.with_suffix(".png")
        report_path = csv_path.with_suffix(".txt")

        # Console output from here on is also the .txt report.
        tee = StdoutTee(sys.stdout, report_path)
        sys.stdout = tee
        try:
            n_parts = len(channel_sets) if channel_sets else 1
            capacity = estimate_capacity(
                rate, args.duration, n_parts, args.guard if channel_sets else 0.0
            )
            recorder = AllanRecorder(csv_path, kept_channels, capacity)
            session = FeedSession(
                dut.client,
                callbacks_feeddata=[recorder],
                device_info={
                    "FirmwareRevision": info.firmware_rev,
                    "ADCConfig": info.adc_config,
                    "TxPowerLevel": tx_power_dbm,
                },
            )
            try:
                await session.start()
                deadline = time.monotonic() + 10.0
                while recorder.n_samples == 0:
                    if time.monotonic() > deadline:
                        raise AllanError("no ADC feed data within 10 s of subscribing")
                    await asyncio.sleep(0.05)
                print(f"Feed running at {rate} SPS. Capture -> {csv_path}")

                windows = []
                if channel_sets is None:
                    print(
                        f"capturing {args.duration:g} s on ch "
                        f"{','.join(map(str, kept_channels))} (passive)"
                    )
                    await capture_for(
                        recorder,
                        args.duration,
                        f"ch {','.join(map(str, kept_channels))}",
                    )
                    print(f"  done: {recorder.n_samples:,} samples")
                    windows.append(
                        {
                            "i0": 0,
                            "i1": recorder.n_samples,
                            "dut_channels": kept_channels,
                            "temp_c": None,
                        }
                    )
                else:
                    for set_idx, board_chs in enumerate(channel_sets):
                        await asyncio.to_thread(
                            cal.set_channels, {c: 0 for c in board_chs}
                        )
                        await asyncio.sleep(args.guard)
                        i0 = recorder.n_samples
                        dut_chs = tuple(c - 1 for c in board_chs)
                        print(
                            f"part {set_idx + 1}/{len(channel_sets)}: "
                            f"ch {','.join(map(str, dut_chs))} shorted, "
                            f"capturing {args.duration:g} s"
                        )
                        await capture_for(
                            recorder,
                            args.duration,
                            f"ch {','.join(map(str, dut_chs))}",
                        )
                        i1 = recorder.n_samples
                        temp_c = await asyncio.to_thread(cal.read_temperature)
                        print(f"  done: {i1 - i0:,} samples, cal board {temp_c:.2f} C")
                        windows.append(
                            {
                                "i0": i0,
                                "i1": i1,
                                "dut_channels": dut_chs,
                                "temp_c": temp_c,
                            }
                        )
                await session.stop()
            finally:
                await session.stop()  # idempotent
                if cal is not None:
                    cal.close()

            traces = reduce_traces(recorder, windows, rate, volts_per_count_ch)
            if not traces:
                raise AllanError("nothing left to plot — every channel railed")

            for t in traces:
                i = int(np.argmin(t.adev_nv))
                print(
                    f"ch{t.dut_ch}: {t.adev_nv[0]:.0f} nV at 1 sample; "
                    f"valley {t.adev_nv[i]:.1f}±{t.err_nv[i]:.1f} nV "
                    f"at τ={t.taus_s[i]:.3g} s"
                )
            for t in traces:
                print(allan_math.format_peaks(t.peaks, t.nv_per_count, f"ch{t.dut_ch}"))

            if channel_sets is None:
                mode_line = f"passive, {args.duration:g} s"
            else:
                temps = ", ".join(f"{w['temp_c']:.2f} C" for w in windows)
                mode_line = (
                    f"cal-board shorts ({cal.fw_id} uid {cal_uid}), "
                    f"{args.duration:g} s/part, board temp: {temps}"
                )
            meta_lines = [
                (
                    f"Allan deviation — {dut.device_name} {dut.client.address} — "
                    f"{info.board_model} @ {rate} SPS"
                ),
                (
                    f"{started:%Y-%m-%d %H:%M}Z · {mode_line} · TX {tx_power_dbm} dBm · "
                    f"{SCRIPT_VERSION} · {info.firmware_rev}"
                ),
            ]
            adev_path, spec_path, psd_path = make_plots(
                traces, meta_lines, png_path, args.no_show
            )
            print(f"Plots saved to {adev_path}")
            print(f"             {spec_path}")
            print(f"             {psd_path}")
            return 0
        except (AllanError, KvsError, CalBoardError) as e:
            # keep the cause in the report; main() will not reprint it
            e.already_reported = True
            print(f"error: {e}")
            raise
        finally:
            sys.stdout = tee.stream
            tee.close()


def main() -> None:
    # Same constraint as board_calibration.py: the feed pump relies on task
    # cancellation that 3.11's asyncio.wait_for can swallow.
    assert sys.version_info >= (
        3,
        12,
    ), f"Python >= 3.12 required, running {sys.version.split()[0]}"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        help="BLE address of the DUT (default: auto-detect, only one may be in range)",
    )
    parser.add_argument(
        "--board",
        choices=sorted(BOARD_MODELS),
        help="board model (default: read from the provisioned Factory namespace)",
    )
    parser.add_argument(
        "--tx-power",
        type=int,
        choices=[-12, -9, -6, -3, 0, 3, 6, 9],
        metavar="DBM",
        help="set the DUT's BLE TX power for the capture; persists on the "
        "device until reboot (default: leave as-is)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="capture length in seconds (per part with --use-calboard; "
        "default: %(default)s)",
    )
    parser.add_argument(
        "--channels",
        type=channel_list_arg,
        help="DUT channels to plot, comma list 0-3 (passive mode only; "
        "default: all four)",
    )
    parser.add_argument(
        "--use-calboard",
        action="store_true",
        help="short inputs through the calibration board (relay 0 position) "
        "to measure the chain noise floor",
    )
    parser.add_argument(
        "--calboard-channels",
        type=channel_list_arg,
        help="DUT channels to short, comma list 0-3, at most one per bridge "
        "(default with --use-calboard: all four, in two parts)",
    )
    parser.add_argument(
        "--cal-port",
        help="serial port of the calibration board (default: probe plausible ports)",
    )
    parser.add_argument(
        "--guard",
        type=float,
        default=DEFAULT_GUARD_S,
        help="settling guard after each relay command, seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help="directory for CSV captures and plots (default: %(default)s)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the PNG but do not open the plot window",
    )
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(run(args)))
    except (AllanError, KvsError, CalBoardError) as e:
        if not getattr(e, "already_reported", False):
            print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
