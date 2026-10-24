import pytest

import cal_math

PHASES = ((1, 3), (2, 4))

# Synthetic "known good" run: 2e6 counts per mV/V chain, 845-count offset.
SYNTH_OFFSET = 845.0
SYNTH_COUNTS_PER_MVV = 2.0e6
SYNTH_STD = 5.0


def make_segments(
    offset=SYNTH_OFFSET,
    cmvv=SYNTH_COUNTS_PER_MVV,
    std=SYNTH_STD,
    n=1000,
    missing=0,
    scramble=False,
    temp=23.0,
    flat=(),
    ch_std=None,
    ch_std_at=None,
):
    """A full synthetic run: 2 phases x 9 sweep states, noise-free means.

    flat: DUT channels that ignore the stimulus (dead channel). ch_std:
    per-DUT-channel window std overrides. ch_std_at: {(dut_ch, phase, seq_idx):
    std} for a single window.
    """
    setpoints = cal_math.ladder_setpoints_mv_per_v()
    segments = []
    for phase, channels in enumerate(PHASES):
        for seq_idx, mv in enumerate(cal_math.SWEEP_SEQUENCE_MV):
            cfg = cal_math.MV_TO_CONFIG[mv]
            if scramble and mv == 10:
                cfg = cal_math.CFG_POS_MID  # +FS reads like +mid
            means = [None] * 4
            for cal_ch in channels:
                dut_ch = cal_ch - 1
                means[dut_ch] = (
                    offset if dut_ch in flat else offset + cmvv * setpoints[cfg]
                )
            stds = []
            for ch in range(4):
                value = (ch_std or {}).get(ch, std)
                value = (ch_std_at or {}).get((ch, phase, seq_idx), value)
                stds.append(value)
            segments.append(
                cal_math.SegmentResult(
                    phase=phase,
                    seq_idx=seq_idx,
                    commanded_mv=mv,
                    cal_channels=channels,
                    t_cmd_unix=0.0,
                    ssn_start=0,
                    ssn_end=n - 1,
                    n_samples=n,
                    missing=missing,
                    means=tuple(means),
                    stds=tuple(stds),
                    temp_c=temp,
                )
            )
    return segments


def synth_expected_span():
    return [SYNTH_COUNTS_PER_MVV * 2 * cal_math.ladder_setpoints_mv_per_v()[0]] * 4


def synth_noise(factor=3.0, tail=1.6, median=SYNTH_STD):
    """A noise budget scaled to the synthetic run: the clean run sits at 1.0x
    its reference, so the ceiling is `factor` x the run's own std."""
    return cal_math.NoiseSpec(
        (median,) * 4, (factor,) * 4, (tail,) * 4, "synthetic"
    )


def gate(segments, noise=None, **overrides):
    params = cal_math.GateParams(noise=noise or synth_noise(), **overrides)
    values = cal_math.collect_config_values(segments)
    return cal_math.gate_run(segments, values, synth_expected_span(), params)


# -- setpoints ---------------------------------------------------------------


def test_setpoints_nominal():
    sp = cal_math.ladder_setpoints_mv_per_v()
    assert sp[0] == pytest.approx(1000.0 * 40 / 20040)
    assert sp[1] == pytest.approx(1000.0 * 20 / 20040)
    assert sp[2] == 0.0  # dead short: exact, resistor-independent
    assert sp[3] == pytest.approx(-sp[1])
    assert sp[4] == pytest.approx(-sp[0])


def test_setpoints_track_resistor_drift():
    sp = cal_math.ladder_setpoints_mv_per_v()
    # A heavier top 10k dilutes every setpoint proportionally.
    bigger = cal_math.ladder_setpoints_mv_per_v(
        (20000.0,) + tuple(cal_math.NOMINAL_LADDER_RESISTORS[1:])
    )
    assert bigger[0] == pytest.approx(1000.0 * 40 / 30040)
    assert bigger[0] < sp[0]


def test_setpoints_reject_bad_resistors():
    with pytest.raises(ValueError):
        cal_math.ladder_setpoints_mv_per_v((10000.0, 0.0, 10.0, 10.0, 10.0, 10000.0))
    with pytest.raises(ValueError):
        cal_math.ladder_setpoints_mv_per_v((10000.0, 10.0))


# -- sweep plan --------------------------------------------------------------


def test_sweep_sequence_covers_configs():
    seq = cal_math.SWEEP_SEQUENCE_MV
    assert seq[0] == 0 and seq[-1] == 0  # zero-anchored
    visits = {}
    for mv in seq:
        visits[cal_math.MV_TO_CONFIG[mv]] = visits.get(cal_math.MV_TO_CONFIG[mv], 0) + 1
    assert visits == {0: 1, 1: 2, 2: 3, 3: 2, 4: 1}  # FS once, mid twice, zero 3x


# -- segment stats -----------------------------------------------------------


def test_segment_stats():
    rows = [(i, 0, 1 + i, 10.0, -5.0, 100) for i in range(3)]  # ch0 = 1,2,3
    means, stds = cal_math.segment_stats(rows)
    assert means[0] == pytest.approx(2.0)
    assert stds[0] == pytest.approx(1.0)
    assert means[1] == pytest.approx(10.0)
    assert stds[1] == pytest.approx(0.0)


def test_segment_stats_empty():
    means, stds = cal_math.segment_stats([])
    assert means == (None,) * 4 and stds == (None,) * 4


# -- gates -------------------------------------------------------------------


def test_gate_clean_run_passes():
    report = gate(make_segments())
    assert report.failures == []


def test_gate_detects_scrambled_order():
    messages = gate(make_segments(scramble=True)).messages
    assert any("gap" in m for m in messages)


def test_gate_detects_pass_delta():
    segments = make_segments()
    for seg in segments:
        if seg.commanded_mv == 5 and seg.seq_idx == 3:  # second +5 pass
            seg.means = tuple(None if m is None else m + 1000.0 for m in seg.means)
    messages = gate(segments).messages
    assert any("pass delta" in m for m in messages)


def test_gate_detects_zero_drift():
    segments = make_segments()
    for seg in segments:
        if seg.commanded_mv == 0 and seg.seq_idx == 8:  # last zero
            seg.means = tuple(None if m is None else m + 500.0 for m in seg.means)
    messages = gate(segments).messages
    assert any("zero spread" in m for m in messages)


def test_gate_detects_wrong_span():
    messages = gate(make_segments(cmvv=1.5e6)).messages  # 25% off the nominal chain
    assert any("span" in m for m in messages)


def test_gate_detects_noisy_window():
    # 10x the run's own reference: the level gate (3x) fires.
    messages = gate(make_segments(std=50.0), noise=synth_noise(median=5.0)).messages
    assert any("window std" in m for m in messages)


def test_gate_detects_single_disturbed_window():
    # Median stays at the reference; one window is 10x the rest. The level
    # gate is blind to that by design; the tail gate must catch it.
    segments = make_segments(ch_std_at={(0, 0, 2): 50.0})
    messages = gate(segments).messages
    assert any("window tail" in m or "is 10.0x this run's median" in m for m in messages)


def test_gate_tail_gate_ignores_undriven_window():
    # ch1 is not driven in phase 0: a huge std there is the open input, not
    # that channel's noise, and must not fail the run.
    segments = make_segments(ch_std_at={(1, 0, 0): 9000.0})
    assert gate(segments).messages == []


def test_gate_detects_empty_window():
    messages = gate(make_segments(n=0)).messages
    assert any("samples in window" in m for m in messages)


def test_gate_detects_dropped_samples():
    messages = gate(make_segments(missing=3)).messages
    assert any("dropped samples" in m for m in messages)


def test_gate_detects_temp_spread():
    segments = make_segments()
    segments[0].temp_c = 25.0  # 2 C above the rest of the run
    messages = gate(segments).messages
    assert any("temp spread" in m for m in messages)


def test_detail_names_dut_and_cal_channels():
    segments = make_segments(ch_std_at={(0, 0, 2): 50.0})
    messages = gate(segments).messages
    assert any("dut ch0 (cal ch1)" in m for m in messages)
    segments = make_segments(std=50.0)
    messages = gate(segments, noise=synth_noise(median=5.0)).messages
    assert any("dut ch3" in m for m in messages)


# -- failure summary -----------------------------------------------------------


def headlines(report):
    """The per-channel verdict lines, without the indented evidence lines."""
    return [line for line in report.summary if not line.startswith(" ")]


def test_summary_clean_run_all_ok():
    report = gate(make_segments())
    assert headlines(report) == [f"ch{i}: OK" for i in range(4)]


def test_summary_ok_channel_prints_noise_line():
    # A passing channel still reports what it measured: that line is the only
    # record of how the accepted ch3 defect did on this unit.
    report = gate(make_segments())
    assert any(
        line.strip().startswith("noise: median std 5 counts") for line in report.summary
    )


def test_summary_noise_line_reports_sem():
    report = gate(make_segments(ch_std={0: 200.0}), noise=synth_noise())
    line = next(line for line in report.summary if "noise: median std" in line)
    # 200 counts over n=1000 -> SEM 6.3
    assert "SEM 6" in line


def test_summary_names_flat_channel():
    report = gate(make_segments(flat=(2,)))
    assert "ch2: NO RESPONSE TO STIMULUS (gross)" in report.summary
    # The wiring pointer names the cal-board channel driving DUT ch2.
    assert any("cal ch3, phase 0" in line for line in report.summary)
    # The measured-vs-expected magnitude is on a sub-line.
    assert any("span 0 counts vs expected ~8.0M" in line for line in report.summary)
    # Other channels are unaffected.
    assert "ch0: OK" in report.summary


def test_summary_noisy_channel_marginal():
    # 5x the 5-count reference = 1.67x the 3x ceiling: over, but within 2x.
    report = gate(make_segments(ch_std={3: 25.0}))
    assert "ch3: NOISY (marginal)" in report.summary
    # ch3 is driven in phase 1 only: 9 windows.
    assert any("9 windows" in line for line in report.summary)


def test_summary_noisy_channel_gross():
    report = gate(make_segments(ch_std={1: 50.0}))  # 10x reference
    assert "ch1: NOISY (gross)" in report.summary


def test_summary_span_off_headline():
    report = gate(make_segments(cmvv=1.5e6))  # 25% off the nominal chain
    assert "ch0: SPAN OFF (gross)" in report.summary
    assert any("(25% off)" in line for line in report.summary)


def test_summary_temp_spread_is_run_level():
    segments = make_segments()
    segments[0].temp_c = 25.0  # 2 C above the rest of the run
    report = gate(segments)
    assert any(line.startswith("run: cal board temp spread") for line in report.summary)
    assert all(f"ch{i}: OK" in report.summary for i in range(4))


# -- noise budget -------------------------------------------------------------


def test_noise_group_lookup_defaults_to_lite():
    assert cal_math.noise_group_for("v700P") == "pro"
    assert cal_math.noise_group_for("v700L") == "lite"
    # Uncharacterized and pre-V7 boards assume the conservative row.
    assert cal_math.noise_group_for("v300") == "lite"
    assert cal_math.noise_group_for("v8l") == "lite"


def test_noise_budget_exceeds_adc_floor():
    """Every budget must sit above the ADC's own noise, or the gate is
    unpassable: this is the check that would have caught the 150-count
    default, which was 0.87x the Lite's silicon floor."""
    floors = {  # counts at 1 ksps/OSR 4096, VREF 1.2, from the datasheet
        1: 23.6,  # PGA 1: 3.38 uV
        32: 172.2,  # PGA >=32: 0.77 uV
    }
    for group, floor_counts in (("pro", floors[1]), ("lite", floors[32])):
        spec = cal_math.NoiseSpec.for_group(group)
        for dut_ch in range(4):
            assert spec.ceiling(dut_ch) > 1.2 * floor_counts, (group, dut_ch)


def test_noise_budget_passes_measured_boards():
    """The reference medians are the measured values, so a good board of each
    family sits at 1.0x and passes with room."""
    for group in ("pro", "lite"):
        spec = cal_math.NoiseSpec.for_group(group)
        for dut_ch in range(4):
            assert spec.medians[dut_ch] <= spec.ceiling(dut_ch)


def test_noise_budget_lite_ch3_is_the_loosest_gate():
    """The V7L ch3 waiver must be visible in the table, not buried: ch3's
    ceiling exceeds every sibling's by a wide margin."""
    spec = cal_math.NoiseSpec.for_group("lite")
    siblings = [spec.ceiling(ch) for ch in range(3)]
    assert spec.ceiling(3) > 4 * max(siblings)


def test_noise_budget_pro_rejects_lite_grade_noise():
    """The point of per-model thresholds: a Pro whose noise is as bad as a
    Lite's (measured in the Pro's own counts) must fail."""
    spec = cal_math.NoiseSpec.for_group("pro")
    # 194 counts on a Lite is 77 ppm of span; the same ppm on a Pro's
    # 12.77M-count span is ~990 counts.
    assert 990 > spec.ceiling(0)


def test_noise_budget_uniform_override_disables_tail_gate():
    spec = cal_math.NoiseSpec.uniform(150.0)
    assert spec.ceiling(0) == 150.0
    assert spec.tail_ratio(0) == float("inf")
    assert spec.group == "override"


def test_noise_spec_rejects_unknown_group():
    with pytest.raises(ValueError):
        cal_math.NoiseSpec.for_group("v9")


def test_window_stats_only_counts_driven_windows():
    segments = make_segments()
    # ch0 is driven in phase 0 only, ch1 in phase 1 only.
    assert len(cal_math.window_stats(segments, 0)) == 9
    assert len(cal_math.window_stats(segments, 1)) == 9
    assert all(seg.n_samples == 1000 for seg, _ in cal_math.window_stats(segments, 0))


def test_median_noise_is_the_reference_measurement():
    assert cal_math.median_noise(make_segments(), 0) == pytest.approx(SYNTH_STD)
    assert cal_math.median_noise(make_segments(ch_std={2: 40.0}), 2) == pytest.approx(
        40.0
    )


def test_sem_scales_with_window_length():
    assert cal_math.sem_counts(200.0, 1000) == pytest.approx(6.324, abs=1e-3)
    assert cal_math.sem_counts(200.0, None) is None
    assert cal_math.sem_counts(200.0, 0) is None


# -- flash entries -----------------------------------------------------------


def test_build_flash_entries_roundtrip():
    segments = make_segments()
    readings = cal_math.final_readings(cal_math.collect_config_values(segments))
    entries = cal_math.build_flash_entries(
        readings,
        "calboard-fw 1.0.0",
        now=__import__("datetime").datetime(
            2026, 8, 7, tzinfo=__import__("datetime").timezone.utc
        ),
        tool="board_calibration 1.0",
        origin="factory",
        adc_gains=[1, 1, 1, 1],
        temp_dut_c=-999.0,
        temp_calboard_c=23.4375,
        noise_stds={0: 51.0, 1: 53.0, 2: 53.0, 3: 1542.0},
        tx_power_dbm=-12,
    )
    assert set(entries) == {
        *(f"ch{i}.{k}" for i in range(4) for k in ("r", "raw", "noise")),
        "cal.date",
        "cal.board",
        "cal.r.prov",
        "cal.temp",
        "cal.tool",
        "cal.origin",
        "cal.adc",
        "cal.txpwr",
    }
    raw = [float(v) for v in entries["ch0.raw"].split(",")]
    assert len(raw) == cal_math.CAL_POINT_COUNT
    assert raw[2] == pytest.approx(SYNTH_OFFSET)
    assert raw[0] > raw[1] > raw[2] > raw[3] > raw[4]
    assert entries["ch0.r"] == "10000,10,10,10,10,10000"
    assert entries["ch3.noise"] == "1542"
    assert entries["cal.txpwr"] == "-12"
    assert entries["cal.r.prov"] == "nominal"
    assert entries["cal.temp"] == "-999,23.4375"  # placeholder DUT temp
    assert entries["cal.date"].startswith("2026-08-07T")


def test_build_flash_entries_without_noise_or_tx_power():
    """Both are optional: a caller that has neither omits the keys rather
    than writing a misleading zero."""
    segments = make_segments()
    readings = cal_math.final_readings(cal_math.collect_config_values(segments))
    entries = cal_math.build_flash_entries(
        readings,
        "calboard-fw 1.0.0",
        tool="t",
        origin="factory",
        adc_gains=[1, 1, 1, 1],
        temp_dut_c=-999.0,
        temp_calboard_c=23.0,
    )
    assert not any(key.endswith(".noise") for key in entries)
    assert "cal.txpwr" not in entries


def test_build_flash_entries_provenance_keys():
    segments = make_segments()
    readings = cal_math.final_readings(cal_math.collect_config_values(segments))
    entries = cal_math.build_flash_entries(
        readings,
        "calboard-fw 1.0.0",
        tool="board_calibration 1.0",
        origin="factory",
        adc_gains=[1, 1, 1, 1],
        temp_dut_c=-999.0,
        temp_calboard_c=23.0,
    )
    assert entries["cal.tool"] == "board_calibration 1.0"
    assert entries["cal.origin"] == "factory"
    assert entries["cal.adc"] == "1,1,1,1"
    # All within the KVS limits (keys <= 15 chars, values <= 128).
    for key, value in entries.items():
        assert len(key) <= 15
        assert len(value) <= 128
