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
):
    """A full synthetic run: 2 phases x 9 sweep states, noise-free means.

    flat: DUT channels that ignore the stimulus (dead channel). ch_std:
    per-DUT-channel window std overrides.
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
                    stds=tuple((ch_std or {}).get(ch, std) for ch in range(4)),
                    temp_c=temp,
                )
            )
    return segments


def synth_expected_span():
    return [SYNTH_COUNTS_PER_MVV * 2 * cal_math.ladder_setpoints_mv_per_v()[0]] * 4


def gate(segments, **overrides):
    params = cal_math.GateParams(**overrides)
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
    messages = gate(make_segments(std=500.0)).messages
    assert any("window std" in m for m in messages)


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
    messages = gate(make_segments(std=500.0)).messages
    assert any("dut ch0 (cal ch1)" in m for m in messages)
    assert any("dut ch3 (cal ch4)" in m for m in messages)


# -- failure summary -----------------------------------------------------------


def test_summary_clean_run_all_ok():
    report = gate(make_segments())
    assert report.summary == [f"ch{i}: OK" for i in range(4)]


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
    report = gate(make_segments(ch_std={3: 150.0}))  # 1.5x the ceiling
    assert "ch3: NOISY (marginal)" in report.summary
    # ch3 is driven in phase 1 only: 9 windows.
    assert any("over in 9 of 9 windows" in line for line in report.summary)


def test_summary_noisy_channel_gross():
    report = gate(make_segments(ch_std={1: 500.0}))
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
        temp_dut_c=-999.0,
        temp_calboard_c=23.4375,
    )
    assert set(entries) == {
        *(f"ch{i}.{k}" for i in range(4) for k in ("r", "raw")),
        "cal.date",
        "cal.board",
        "cal.r.prov",
        "cal.temp",
    }
    raw = [float(v) for v in entries["ch0.raw"].split(",")]
    assert len(raw) == cal_math.CAL_POINT_COUNT
    assert raw[2] == pytest.approx(SYNTH_OFFSET)
    assert raw[0] > raw[1] > raw[2] > raw[3] > raw[4]
    assert entries["ch0.r"] == "10000,10,10,10,10,10000"
    assert entries["cal.r.prov"] == "nominal"
    assert entries["cal.temp"] == "-999,23.4375"  # placeholder DUT temp
    assert entries["cal.date"].startswith("2026-08-07T")


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

    # Absent provenance: no keys stamped (older readers see the old layout).
    bare = cal_math.build_flash_entries(
        readings, "x", temp_dut_c=-999.0, temp_calboard_c=23.0
    )
    for key in ("cal.tool", "cal.origin", "cal.adc"):
        assert key not in bare
