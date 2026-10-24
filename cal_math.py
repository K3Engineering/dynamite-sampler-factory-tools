"""Calibration math for the factory board calibration.

Pure functions and data — no I/O, no BLE, no serial. The storage format this
produces is specified in docs/flash-schema-v2.md (repo root).

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

from dynamite_sampler.cal_math import (
    CAL_POINT_COUNT,
    NOMINAL_LADDER_RESISTORS,
    expected_counts_per_mvv,
    ladder_setpoints_mv_per_v,
)

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
# exists (see docs/flash-schema-v2.md).
PROVENANCE_NOMINAL = "nominal"

# -- noise budget -------------------------------------------------------------
#
# The window-std gate is in ADC counts, and a count means very different
# voltages on different board families: a v700P has afe_gain 101 and 4.53 V of
# excitation (one count = 1.4 nV, span ~12.7 M counts), a v700L has afe_gain
# 1.0 and 2.8 V with the ADC's own PGA at 32 (one count = 4.5 nV, span
# ~2.5 M counts). The ADS131M04's own noise at 1 ksps/OSR 4096 is 23.6 counts
# at PGA 1 and 172 counts at PGA 32, so a single absolute ceiling is either
# 6x the Pro's noise floor or *below* the Lite's silicon floor.
#
# So: a reference window-std median per channel (a measured fact about a
# known-good board), a policy factor (how much worse we ship), and a
# within-run tail ratio (max window vs that channel's own median window, which
# catches a disturbed measurement without fighting the level gate).
#
# The per-window std is itself noisy (~2.2% estimation floor at n=1000, plus
# real drift), and on a burst-dominated channel it is worse: ch3 of the Lite
# shows ~40 RFI bursts per window, so its std carries the Poisson scatter of
# the event count (~16%). That is why the level gate uses the median of the
# driven windows rather than the max, and the tail gate compares the max to
# the same run's median instead of to an absolute number.
NOISE_MEDIANS = {
    # Driven-channel window std medians on known-good boards, counts
    # (1 s windows, 1 ksps). Provisional: one board each so far.
    "pro": (51.0, 53.0, 53.0, 54.0),  # v700P: 4.0-4.3 ppm of span
    "lite": (194.0, 197.0, 212.0, 1542.0),  # v700L: 77/78/84 ppm; ch3 self-RFI
}

NOISE_FACTORS = {
    "pro": (3.0, 3.0, 3.0, 3.0),
    # ch3 is the accepted V7L self-RFI defect: it must ship. TODO verify: the
    # ceiling is compared to the run median, so the cited max (2177 counts on
    # 2026-10-22) does not justify 2.3 — as a max it is 1.41x reference; as a
    # median it would clear a much tighter factor. It might be a median-to-
    # median spread across runs that 2.3 actually covers.
    "lite": (3.0, 3.0, 3.0, 2.3),
}

NOISE_TAIL_RATIOS = {
    "pro": (1.6, 1.6, 1.6, 1.6),
    # ch3's std estimate is burst-count noisy; a 2.6 gate on max/median gives
    # ~0 false failures on the measured spread (p99.9 of max/median is 2.1).
    "lite": (1.6, 1.6, 1.6, 2.6),
}

# board_model -> noise group. Anything unlisted assumes the lite row (the
# conservative choice for pre-V7 boards, whose noise was never characterized).
NOISE_GROUPS = {"v700P": "pro", "v600L": "lite", "v700L": "lite"}
DEFAULT_NOISE_GROUP = "lite"


def noise_group_for(board_model):
    """The noise-budget group a board model belongs to (lazy default: lite)."""
    return NOISE_GROUPS.get(board_model, DEFAULT_NOISE_GROUP)


@dataclass(frozen=True)
class NoiseSpec:
    """Per-channel noise budget: ceiling = medians[ch] * factors[ch], and a
    within-run max/median tail ratio."""

    medians: tuple
    factors: tuple
    tail_ratios: tuple
    group: str = "override"

    def ceiling(self, dut_ch):
        return self.medians[dut_ch] * self.factors[dut_ch]

    def tail_ratio(self, dut_ch):
        return self.tail_ratios[dut_ch]

    @classmethod
    def uniform(cls, counts, group="override"):
        """One flat ceiling for every channel (bring-up escape hatch).

        The tail gate is disabled: with no per-channel reference there is
        nothing to compare a run's own median against except itself."""
        return cls((float(counts),) * 4, (1.0,) * 4, (float("inf"),) * 4, group)

    @classmethod
    def for_group(cls, group):
        if group not in NOISE_MEDIANS:
            raise ValueError(f"unknown noise group {group!r}")
        return cls(
            NOISE_MEDIANS[group], NOISE_FACTORS[group], NOISE_TAIL_RATIOS[group], group
        )


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
    # Per-channel noise budget. Default is the "lite" group, the conservative
    # choice for a board whose noise was never characterized; the driver
    # always passes the board's own group (or a uniform override).
    noise: NoiseSpec = None  # type: ignore[assignment]
    min_gap_counts: float = 1000.0  # monotonic spacing between config means
    pass_tol_counts: float = 200.0  # |pass1 - pass2| on the interior points
    zero_tol_counts: float = 200.0  # spread of the three zero visits
    span_tol: float = 0.10  # |measured/expected - 1| on the +/-FS span
    max_temp_spread_c: float = 0.2  # cal board temp drift over the run

    def __post_init__(self):
        if self.noise is None:
            self.noise = NoiseSpec.for_group(DEFAULT_NOISE_GROUP)


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


def window_stats(segments, dut_ch):
    """That channel's driven windows as [(segment, std)], in run order.

    Only windows where the channel was actually driven are noise evidence; an
    undriven channel can be wild (a Lite channel on an open input reads ~6.8k
    counts of mains pickup) and says nothing about its own noise floor."""
    cal_ch = dut_ch + 1
    return [
        (seg, seg.stds[dut_ch])
        for seg in segments
        if cal_ch in seg.cal_channels and seg.stds[dut_ch] is not None
    ]


def noise_stds(segments, dut_ch):
    """That channel's driven-window stds, in run order (see window_stats)."""
    return [std for _, std in window_stats(segments, dut_ch)]


def median_noise(segments, dut_ch):
    """The channel's reference measurement: median std of its driven windows."""
    stds = noise_stds(segments, dut_ch)
    return statistics.median(stds) if stds else None


def sem_counts(std, n):
    """Standard error of the window mean: what the noise costs the stored
    reading. Assumes independent samples — true for thermal noise, optimistic
    for a burst-dominated channel (ch3's ~40 events per window make its honest
    SEM ~5x larger). Printed, never gated."""
    return None if not n else std / (n**0.5)


def _noise_failures(segments, dut_ch, label_for, params):
    """Level + tail gates for one driven channel.

    Level: the median window std over the driven windows vs the channel's
    budget (reference median x factor). The median is ~3x steadier run to run
    than a single window, so this gate is the tight one.
    Tail: the worst window vs that same median, within this run. Catches a
    single disturbed window, which the level gate is blind to by design.
    """
    stats = window_stats(segments, dut_ch)
    if not stats:
        return []
    stds = [std for _, std in stats]
    med = statistics.median(stds)
    worst_seg, worst = max(stats, key=lambda pair: pair[1])
    ceiling = params.noise.ceiling(dut_ch)
    out = []
    if med > ceiling:
        out.append(
            GateFailure(
                "window-std",
                f"dut ch{dut_ch}: median window std {med:.1f} counts "
                f"vs {ceiling:.0f} ceiling "
                f"({med / params.noise.medians[dut_ch]:.1f}x reference, "
                f"{len(stds)} windows)",
                (dut_ch,),
            )
        )
    ratio = params.noise.tail_ratio(dut_ch)
    if med > 0 and worst > ratio * med:
        out.append(
            GateFailure(
                "window-tail",
                f"{label_for(worst_seg)}: dut ch{dut_ch} "
                f"(cal ch{dut_ch + 1}) window std {worst:.1f} counts is "
                f"{worst / med:.1f}x this run's median {med:.0f} "
                f"(> {ratio:.1f}x) — one disturbed window",
                (dut_ch,),
            )
        )
    return out


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

    def label_for(seg):
        return f"phase {seg.phase} seq {seg.seq_idx} ({seg.commanded_mv:+d} mV)"

    for dut_ch in range(4):
        failures.extend(_noise_failures(segments, dut_ch, label_for, params))

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
    reached the ADC. A passing channel still reports its noise line: the numbers are the
    evidence the budget was met, and the only record of how the accepted ch3 defect
    actually measured on this unit.
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


def _noise_line(dut_ch, segments, params):
    """One channel's noise measurement in counts, with its SEM.

    SEM is what the noise costs the stored reading (the gate stores window
    means, not samples). TODO verify: ppm of span may belong in the line as
    well, putting boards with different gain chains on one scale, since a
    count is 5.1x more span on a Lite (0.078 ppm on a Pro against 0.400) —
    not the 19x that the measured noise medians (78 against 4.0 ppm) suggest.
    """
    stats = window_stats(segments, dut_ch)
    if not stats:
        return None
    med = statistics.median(std for _, std in stats)
    ns = [seg.n_samples for seg, _ in stats]
    sem = sem_counts(med, int(statistics.median(ns)))
    ceiling = params.noise.ceiling(dut_ch)
    ref = params.noise.medians[dut_ch]
    extra = f", SEM {sem:.0f}" if sem is not None else ""
    return (
        f"     noise: median std {med:.0f} counts ({med / ref:.1f}x "
        f"{params.noise.group} reference {ref:.0f}, ceiling {ceiling:.0f}, "
        f"worst {max(std for _, std in stats):.0f}{extra}; {len(stats)} windows)"
    )


def _channel_verdict(
    dut_ch, segments, config_values, expected_span_by_ch, params, failures
):
    """One channel's headline verdict plus indented explanation sub-lines."""
    ch_failures = [f for f in failures if dut_ch in f.channels]
    if not ch_failures:
        line = _noise_line(dut_ch, segments, params)
        return [f"ch{dut_ch}: OK"] + ([line] if line else [])
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
        med = median_noise(segments, dut_ch)
        # "marginal": the median sits within 2x the ceiling — likely a real
        # noise floor at the threshold, not a broken channel.
        tag = "marginal" if med <= 2 * params.noise.ceiling(dut_ch) else "gross"
        sub = [_noise_line(dut_ch, segments, params), f"     driven by {where}"]
        extra = kinds - {"window-std"}
        if extra:
            sub.append(f"     also: {', '.join(sorted(extra))} — see detail")
        return [f"ch{dut_ch}: NOISY ({tag})", *sub]

    if "window-tail" in kinds:
        # The level gate passed: this run's median is within budget, but one
        # window was disturbed. Worth a rerun, not a hardware verdict.
        failure = next(f for f in ch_failures if f.kind == "window-tail")
        return [
            f"ch{dut_ch}: WINDOW DISTURBED (retry)",
            f"     {failure.message}",
            _noise_line(dut_ch, segments, params),
        ]

    return [f"ch{dut_ch}: FAILED ({', '.join(sorted(kinds))}) — see detail"]


def _fmt(value):
    """Compact float formatting for KVS values ('.9g' round-trips doubles)."""
    return f"{value:.9g}"


def build_flash_entries(
    readings,
    cal_board_id,
    now=None,
    resistors=NOMINAL_LADDER_RESISTORS,
    r_provenance=PROVENANCE_NOMINAL,
    *,
    tool,
    origin,
    adc_gains,
    temp_dut_c,
    temp_calboard_c,
    noise_stds=None,
    tx_power_dbm=None,
):
    """Factory-namespace entries for a passed calibration.

    readings: {dut_ch: [5 config means, storage order]} for all 4 channels.
    tool: host script version (cal.tool). origin: e.g. "factory" (cal.origin).
    adc_gains: per-channel PGA gains at calibration (cal.adc).
    temp_dut_c/temp_calboard_c: temperatures at calibration (cal.temp,
    "dut,calboard"); temp_dut_c is a placeholder until the DUT sensor is
    plumbed.
    noise_stds: {dut_ch: median window std in counts} -> chN.noise, so the
    accepted-defect channels are auditable on the board itself and the noise
    reference medians can be re-derived from the shipped fleet.
    tx_power_dbm: BLE TX power during calibration -> cal.txpwr, because it is
    a measurement condition that moves a self-RFI channel's noise (ch3 on the
    Lite measured 240 counts at 0 dBm and 185 at -12).
    Keys/values are KVS strings; the key/value length limits are enforced by
    the device's KVS namespace writes, not here.
    """
    now = now or datetime.now(timezone.utc)
    entries = {}
    for dut_ch in range(4):
        entries[f"ch{dut_ch}.r"] = ",".join(_fmt(v) for v in resistors)
        entries[f"ch{dut_ch}.raw"] = ",".join(_fmt(v) for v in readings[dut_ch])
        if noise_stds and dut_ch in noise_stds:
            entries[f"ch{dut_ch}.noise"] = _fmt(noise_stds[dut_ch])
    entries["cal.date"] = now.isoformat(timespec="seconds")
    entries["cal.board"] = cal_board_id
    entries["cal.r.prov"] = r_provenance
    entries["cal.temp"] = f"{_fmt(temp_dut_c)},{_fmt(temp_calboard_c)}"
    entries["cal.tool"] = tool
    entries["cal.origin"] = origin
    entries["cal.adc"] = ",".join(_fmt(g) for g in adc_gains)
    if tx_power_dbm is not None:
        entries["cal.txpwr"] = _fmt(tx_power_dbm)
    return entries
