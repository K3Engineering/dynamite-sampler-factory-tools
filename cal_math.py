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
MV_TO_CONFIG = {
    10: CFG_POS_FS,
    5: CFG_POS_MID,
    0: CFG_ZERO,
    -5: CFG_NEG_MID,
    -10: CFG_NEG_FS,
}

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


def expected_span_counts(
    adc_fsr_v, afe_gain, pga_gain, exc_v, resistors=NOMINAL_LADDER_RESISTORS
):
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
    temp_c: float  # cal board TMP118 read at the end of this segment

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
    max_temp_spread_c: float = 0.2  # cal board temp drift over the run


@dataclass
class GateFailure:
    """One gate violation: a machine-readable kind, the human-readable detail
    line, and the DUT channels it implicates (empty = run-level)."""

    kind: str
    message: str
    channels: tuple = ()


@dataclass
class GateReport:
    """gate_run result: the detail failures plus per-channel verdict lines
    for the operator (summary sub-lines are indented)."""

    failures: list
    summary: list

    @property
    def messages(self):
        return [f.message for f in self.failures]


def gate_run(segments, config_values, expected_span_by_ch, params):
    """Run all quality gates. Returns a GateReport (empty failures = pass)."""
    failures = []

    for seg in segments:
        label = f"phase {seg.phase} seq {seg.seq_idx} ({seg.commanded_mv:+d} mV)"
        seg_chs = tuple(cal_ch - 1 for cal_ch in seg.cal_channels)
        if seg.n_samples < params.min_window_samples:
            failures.append(
                GateFailure(
                    "window-samples",
                    f"{label}: only {seg.n_samples} samples in window",
                    seg_chs,
                )
            )
        if seg.missing > params.max_missing:
            failures.append(
                GateFailure(
                    "dropped-samples",
                    f"{label}: {seg.missing} dropped samples in window",
                    seg_chs,
                )
            )
        for cal_ch in seg.cal_channels:
            std = seg.stds[cal_ch - 1]
            if std is not None and std > params.max_std_counts:
                failures.append(
                    GateFailure(
                        "window-std",
                        f"{label}: dut ch{cal_ch - 1} (cal ch{cal_ch}) "
                        f"window std {std:.1f} counts",
                        (cal_ch - 1,),
                    )
                )

    temps = [seg.temp_c for seg in segments]
    spread = max(temps) - min(temps)
    if spread > params.max_temp_spread_c:
        failures.append(
            GateFailure(
                "temp-spread",
                f"cal board temp spread {spread:.2f} C > {params.max_temp_spread_c}",
            )
        )

    for dut_ch in range(4):
        by_config = config_values.get(dut_ch, {})
        missing_cfgs = [k for k in range(CAL_POINT_COUNT) if not by_config.get(k)]
        if missing_cfgs:
            failures.append(
                GateFailure(
                    "no-data",
                    f"dut ch{dut_ch}: no data for configs {missing_cfgs}",
                    (dut_ch,),
                )
            )
            continue
        means = [statistics.fmean(by_config[k]) for k in range(CAL_POINT_COUNT)]

        # Monotonic ordering with a minimum gap (a real ladder spread is
        # millions of counts; a small gap means a dead/wrong channel).
        for k in range(CAL_POINT_COUNT - 1):
            gap = means[k] - means[k + 1]
            if gap < params.min_gap_counts:
                failures.append(
                    GateFailure(
                        "gap",
                        f"dut ch{dut_ch}: {CONFIG_LABELS[k]}/{CONFIG_LABELS[k + 1]} "
                        f"gap {gap:.0f} < {params.min_gap_counts:.0f}",
                        (dut_ch,),
                    )
                )

        # Reversal repeatability: the two passes through each interior point.
        for k in (CFG_POS_MID, CFG_NEG_MID):
            vals = by_config[k]
            if len(vals) == 2:
                delta = abs(vals[0] - vals[1])
                if delta > params.pass_tol_counts:
                    failures.append(
                        GateFailure(
                            "pass-delta",
                            f"dut ch{dut_ch}: {CONFIG_LABELS[k]} pass delta {delta:.0f} "
                            f"> {params.pass_tol_counts:.0f}",
                            (dut_ch,),
                        )
                    )

        # Drift closure: spread of the three dead-short visits.
        zeros = by_config[CFG_ZERO]
        if len(zeros) >= 2:
            zero_spread = max(zeros) - min(zeros)
            if zero_spread > params.zero_tol_counts:
                failures.append(
                    GateFailure(
                        "zero-spread",
                        f"dut ch{dut_ch}: zero spread {zero_spread:.0f} "
                        f"> {params.zero_tol_counts:.0f}",
                        (dut_ch,),
                    )
                )

        # Span cross-check against the nominal chain (gross errors only).
        expected = expected_span_by_ch[dut_ch]
        if expected:
            span = means[CFG_POS_FS] - means[CFG_NEG_FS]
            if abs(span / expected - 1.0) > params.span_tol:
                failures.append(
                    GateFailure(
                        "span",
                        f"dut ch{dut_ch}: span {span:.0f} vs nominal-chain "
                        f"{expected:.0f} (>{params.span_tol:.0%} off)",
                        (dut_ch,),
                    )
                )

    summary = summarize_failures(
        segments, config_values, expected_span_by_ch, params, failures
    )
    return GateReport(failures, summary)


def summarize_failures(segments, config_values, expected_span_by_ch, params, failures):
    """Per-channel verdict lines for the operator, then run-level failures.

    Each channel gets an OK or a named signature (gross/marginal) with the key numbers on indented
    sub-lines. A total response across the sweep below the gap floor means the stimulus never
    reached the ADC.
    """
    lines = []
    for dut_ch in range(4):
        lines.extend(
            _channel_verdict(
                dut_ch, segments, config_values, expected_span_by_ch, params, failures
            )
        )
    lines.extend(f"run: {f.message}" for f in failures if not f.channels)
    return lines


def _channel_verdict(
    dut_ch, segments, config_values, expected_span_by_ch, params, failures
):
    """One channel's headline verdict plus indented explanation sub-lines."""
    ch_failures = [f for f in failures if dut_ch in f.channels]
    if not ch_failures:
        return [f"ch{dut_ch}: OK"]
    kinds = {f.kind for f in ch_failures}
    cal_ch = dut_ch + 1
    driven = [seg for seg in segments if cal_ch in seg.cal_channels]
    where = f"cal ch{cal_ch}" + (f", phase {driven[0].phase}" if driven else "")

    by_config = config_values.get(dut_ch, {})
    if not all(by_config.get(k) for k in range(CAL_POINT_COUNT)):
        missing = [k for k in range(CAL_POINT_COUNT) if not by_config.get(k)]
        return [
            f"ch{dut_ch}: NO DATA (gross)",
            f"     no samples for configs {missing} ({where}) — capture/feed problem",
        ]

    means = [statistics.fmean(by_config[k]) for k in range(CAL_POINT_COUNT)]
    response = max(means) - min(means)
    span = means[CFG_POS_FS] - means[CFG_NEG_FS]
    expected = expected_span_by_ch[dut_ch]

    # Flat channel: the whole +/-FS sweep moved the output less than one
    # minimally-acceptable step — the stimulus never reached the ADC.
    if response < params.min_gap_counts:
        return [
            f"ch{dut_ch}: NO RESPONSE TO STIMULUS (gross)",
            f"     span {span:.0f} counts vs expected ~{expected / 1e6:.1f}M; "
            f"all 5 config means within {response:.0f} counts",
            f"     driven by {where} — check fixture wiring / AFE channel",
        ]

    if "span" in kinds:
        sub = []
        if expected:
            sub.append(
                f"     span {span:.0f} vs expected ~{expected / 1e6:.1f}M counts "
                f"({abs(span / expected - 1.0):.0%} off), {where}"
            )
        extra = kinds - {"span"}
        if extra:
            sub.append(f"     also: {', '.join(sorted(extra))} — see detail")
        return [f"ch{dut_ch}: SPAN OFF (gross)", *sub]

    if "window-std" in kinds:
        stds = [seg.stds[dut_ch] for seg in driven if seg.stds[dut_ch] is not None]
        over = sum(s > params.max_std_counts for s in stds)
        # "marginal": within 2x of the (gross-fault) ceiling — likely a real
        # noise floor sitting at the threshold, not a broken channel.
        tag = "marginal" if max(stds) <= 2 * params.max_std_counts else "gross"
        sub = [
            f"     window std {min(stds):.0f}-{max(stds):.0f} counts vs "
            f"{params.max_std_counts:.0f} ceiling; over in {over} of "
            f"{len(stds)} windows ({where})"
        ]
        extra = kinds - {"window-std"}
        if extra:
            sub.append(f"     also: {', '.join(sorted(extra))} — see detail")
        return [f"ch{dut_ch}: NOISY ({tag})", *sub]

    return [f"ch{dut_ch}: FAILED ({', '.join(sorted(kinds))}) — see detail"]


def _fmt(value):
    """Compact float formatting for KVS values ('.9g' round-trips doubles)."""
    return f"{value:.9g}"


def build_flash_entries(
    readings,
    cal_board_id,
    exc_mv=None,
    now=None,
    resistors=NOMINAL_LADDER_RESISTORS,
    r_provenance=PROVENANCE_NOMINAL,
    tool=None,
    origin=None,
    adc_gains=None,
    *,
    temp_dut_c,
    temp_calboard_c,
):
    """Factory-namespace entries for a passed calibration.

    readings: {dut_ch: [5 config means, storage order]} for all 4 channels.
    tool: host script version (cal.tool). origin: e.g. "factory" (cal.origin).
    temp_dut_c/temp_calboard_c: temperatures at calibration (cal.temp,
    "dut,calboard"); temp_dut_c is a placeholder until the DUT sensor is
    plumbed.
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
    entries["cal.temp"] = f"{_fmt(temp_dut_c)},{_fmt(temp_calboard_c)}"
    if exc_mv is not None:
        entries["cal.exc.mv"] = _fmt(exc_mv)
    if tool is not None:
        entries["cal.tool"] = tool
    if origin is not None:
        entries["cal.origin"] = origin
    if adc_gains is not None:
        entries["cal.adc"] = ",".join(_fmt(g) for g in adc_gains)
    return entries
