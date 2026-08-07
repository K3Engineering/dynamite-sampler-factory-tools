"""Calibration math for the factory board calibration.

Pure functions and data — no I/O, no BLE, no serial. The storage format this
produces is specified in docs/flash-schema-v1.md (repo root).

The stimulus is the calibration board's resistor ladder, powered by the DUT's
own excitation (ratiometric: only resistor ratios matter):

    EXC+ --[R0=10k]--t1--[R1=10R]--t2--[R2=10R]--t3--[R3=10R]--t4--[R4=10R]--t5--[R5=10k]--GND

Five differential configs are measured per channel, in storage order:

    idx 0: (t1,t5)  +full scale   (bridge output "+10 mV")
    idx 1: (t2,t4)  +mid          (bridge output "+5 mV")
    idx 2: (t3,t3)  zero          (dead short — exact, resistor-independent)
    idx 3: (t4,t2)  -mid          (bridge output "-5 mV")
    idx 4: (t5,t1)  -full scale   (bridge output "-10 mV")

The "mV" labels are the cal board firmware's nominal names; the actual
setpoints are computed from the ladder resistors (ladder_setpoints_mv_per_v).
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

ADC_COUNTS_PER_POLARITY = 1 << 23  # 24-bit bipolar

# [top 10k, four 10R, bottom 10k], in signal order from EXC+ to GND.
NOMINAL_LADDER_RESISTORS = (10000.0, 10.0, 10.0, 10.0, 10.0, 10000.0)

CAL_POINT_COUNT = 5
LADDER_RESISTOR_COUNT = 6

CONFIG_LABELS = ("(t1,t5)", "(t2,t4)", "(t3,t3)", "(t4,t2)", "(t5,t1)")

# Config indices by signal role.
CFG_POS_FS = 0
CFG_POS_MID = 1
CFG_ZERO = 2
CFG_NEG_MID = 3
CFG_NEG_FS = 4

# Bridge output label (mV) -> config index (storage order above).
MV_TO_CONFIG = {10: CFG_POS_FS, 5: CFG_POS_MID, 0: CFG_ZERO, -5: CFG_NEG_MID, -10: CFG_NEG_FS}

# Zero-anchored reversal sweep: starts and ends on the dead short (drift
# closure check), interior points visited once per approach direction
# (repeatability + hysteresis probe). ~9 states per phase.
SWEEP_SEQUENCE_MV = (0, 5, 10, 5, 0, -5, -10, -5, 0)

# Provenance tag for the ladder resistors while no per-board characterization
# exists (see docs/flash-schema-v1.md).
PROVENANCE_NOMINAL = "nominal"


def ladder_setpoints_mv_per_v(resistors=NOMINAL_LADDER_RESISTORS):
    """Differential setpoints (mV/V of excitation) per config, storage order.

    Pure function of the ladder resistors — the ladder is ratiometric, so the
    excitation cancels and only ratios matter.
    """
    if len(resistors) != LADDER_RESISTOR_COUNT:
        raise ValueError(f"need {LADDER_RESISTOR_COUNT} ladder resistors")
    if any(r <= 0 for r in resistors):
        raise ValueError("ladder resistors must be positive")
    # Resistance below each tap (toward GND); tap t_k sits above resistors[k].
    below = [0.0] * CAL_POINT_COUNT
    acc = 0.0
    for i in range(LADDER_RESISTOR_COUNT - 1, 0, -1):
        acc += resistors[i]
        below[i - 1] = acc
    total = acc + resistors[0]
    return [
        1000.0 * (below[k] - below[CAL_POINT_COUNT - 1 - k]) / total
        for k in range(CAL_POINT_COUNT)
    ]


def expected_counts_per_mvv(adc_fsr_v, afe_gain, pga_gain, exc_v):
    """Nominal analog chain: ADC counts per mV/V of load-cell output."""
    return ADC_COUNTS_PER_POLARITY * afe_gain * pga_gain / (adc_fsr_v * 1000.0) * exc_v


def expected_span_counts(adc_fsr_v, afe_gain, pga_gain, exc_v,
                         resistors=NOMINAL_LADDER_RESISTORS):
    """Expected count difference between the +FS and -FS configs."""
    sp = ladder_setpoints_mv_per_v(resistors)
    return expected_counts_per_mvv(adc_fsr_v, afe_gain, pga_gain, exc_v) * (
        sp[CFG_POS_FS] - sp[CFG_NEG_FS]
    )


def segment_stats(rows):
    """(means, stds) per channel over rows of (ssn, t_unix_ms, ch0..ch3).

    std is the sample standard deviation (ddof=1). Empty window -> all None.
    """
    n = len(rows)
    if n == 0:
        return (None,) * 4, (None,) * 4
    means, stds = [], []
    for ch in range(4):
        vals = [r[2 + ch] for r in rows]
        means.append(statistics.fmean(vals))
        stds.append(statistics.stdev(vals) if n > 1 else 0.0)
    return tuple(means), tuple(stds)


@dataclass
class SegmentResult:
    """Statistics of one commanded sweep state's valid window."""

    phase: int
    seq_idx: int
    commanded_mv: int
    cal_channels: tuple  # cal-board channels driven in this phase (1-based)
    t_cmd_unix: float
    ssn_start: int | None
    ssn_end: int | None
    n_samples: int
    missing: int  # emitted-but-not-received samples inside the window
    means: tuple  # per DUT channel (4), None when the window was empty
    stds: tuple

    @property
    def config_idx(self):
        return MV_TO_CONFIG[self.commanded_mv]


def collect_config_values(segments):
    """{dut_ch: {config_idx: [window means]}} for driven channels."""
    out = {}
    for seg in segments:
        for cal_ch in seg.cal_channels:
            dut_ch = cal_ch - 1
            value = seg.means[dut_ch]
            if value is None:
                continue
            out.setdefault(dut_ch, {}).setdefault(seg.config_idx, []).append(value)
    return out


def final_readings(config_values):
    """{dut_ch: [mean per config, storage order]}. Call only after gates pass."""
    return {
        dut_ch: [statistics.fmean(by_config[k]) for k in range(CAL_POINT_COUNT)]
        for dut_ch, by_config in config_values.items()
    }


@dataclass
class GateParams:
    """Quality-gate thresholds. Defaults are gross-fault detectors; tune with
    hardware experience (noise of a window mean is ~1 count)."""

    min_window_samples: int = 100  # window must hold at least this many samples
    max_missing: int = 0  # dropped samples tolerated inside a window
    max_std_counts: float = 100.0  # per-window noise ceiling (driven channels)
    min_gap_counts: float = 1000.0  # monotonic spacing between config means
    pass_tol_counts: float = 200.0  # |pass1 - pass2| on the interior points
    zero_tol_counts: float = 200.0  # spread of the three zero visits
    span_tol: float = 0.10  # |measured/expected - 1| on the +/-FS span


def gate_run(segments, config_values, expected_span_by_ch, params):
    """Run all quality gates. Returns a list of failure strings (empty = pass)."""
    failures = []

    for seg in segments:
        label = f"phase {seg.phase} seq {seg.seq_idx} ({seg.commanded_mv:+d} mV)"
        if seg.n_samples < params.min_window_samples:
            failures.append(f"{label}: only {seg.n_samples} samples in window")
        if seg.missing > params.max_missing:
            failures.append(f"{label}: {seg.missing} dropped samples in window")
        for cal_ch in seg.cal_channels:
            std = seg.stds[cal_ch - 1]
            if std is not None and std > params.max_std_counts:
                failures.append(f"{label}: ch{cal_ch - 1} window std {std:.1f} counts")

    for dut_ch in range(4):
        by_config = config_values.get(dut_ch, {})
        missing_cfgs = [k for k in range(CAL_POINT_COUNT) if not by_config.get(k)]
        if missing_cfgs:
            failures.append(f"ch{dut_ch}: no data for configs {missing_cfgs}")
            continue
        means = [statistics.fmean(by_config[k]) for k in range(CAL_POINT_COUNT)]

        # Monotonic ordering with a minimum gap (a real ladder spread is
        # millions of counts; a small gap means a dead/wrong channel).
        for k in range(CAL_POINT_COUNT - 1):
            gap = means[k] - means[k + 1]
            if gap < params.min_gap_counts:
                failures.append(
                    f"ch{dut_ch}: {CONFIG_LABELS[k]}/{CONFIG_LABELS[k + 1]} "
                    f"gap {gap:.0f} < {params.min_gap_counts:.0f}"
                )

        # Reversal repeatability: the two passes through each interior point.
        for k in (CFG_POS_MID, CFG_NEG_MID):
            vals = by_config[k]
            if len(vals) == 2:
                delta = abs(vals[0] - vals[1])
                if delta > params.pass_tol_counts:
                    failures.append(
                        f"ch{dut_ch}: {CONFIG_LABELS[k]} pass delta {delta:.0f} "
                        f"> {params.pass_tol_counts:.0f}"
                    )

        # Drift closure: spread of the three dead-short visits.
        zeros = by_config[CFG_ZERO]
        if len(zeros) >= 2:
            spread = max(zeros) - min(zeros)
            if spread > params.zero_tol_counts:
                failures.append(
                    f"ch{dut_ch}: zero spread {spread:.0f} "
                    f"> {params.zero_tol_counts:.0f}"
                )

        # Span cross-check against the nominal chain (gross errors only).
        expected = expected_span_by_ch[dut_ch]
        if expected:
            span = means[CFG_POS_FS] - means[CFG_NEG_FS]
            if abs(span / expected - 1.0) > params.span_tol:
                failures.append(
                    f"ch{dut_ch}: span {span:.0f} vs nominal-chain {expected:.0f} "
                    f"(>{params.span_tol:.0%} off)"
                )

    return failures


def _fmt(value):
    """Compact float formatting for KVS values ('.9g' round-trips doubles)."""
    return f"{value:.9g}"


def build_flash_entries(readings, cal_board_id, exc_mv=None, now=None,
                        resistors=NOMINAL_LADDER_RESISTORS,
                        r_provenance=PROVENANCE_NOMINAL):
    """Factory-namespace entries for a passed calibration.

    readings: {dut_ch: [5 config means, storage order]} for all 4 channels.
    Keys/values are KVS strings; the key/value length limits are enforced by
    the transport (KvsClient.set), not here.
    """
    now = now or datetime.now(timezone.utc)
    entries = {}
    for dut_ch in range(4):
        entries[f"ch{dut_ch}.r"] = ",".join(_fmt(v) for v in resistors)
        entries[f"ch{dut_ch}.raw"] = ",".join(_fmt(v) for v in readings[dut_ch])
    entries["cal.date"] = now.isoformat(timespec="seconds")
    entries["cal.board"] = cal_board_id
    entries["cal.r.prov"] = r_provenance
    if exc_mv is not None:
        entries["cal.exc.mv"] = _fmt(exc_mv)
    return entries
