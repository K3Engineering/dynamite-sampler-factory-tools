import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import allan_plot
from dynamite_sampler.block import Block


def args(**kw):
    base = {
        "use_calboard": False,
        "channels": None,
        "calboard_channels": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_plan_passive_defaults_to_all_channels():
    channel_sets, kept = allan_plot.plan_capture(args())
    assert channel_sets is None
    assert kept == (0, 1, 2, 3)


def test_plan_passive_channel_subset():
    channel_sets, kept = allan_plot.plan_capture(args(channels=(2,)))
    assert channel_sets is None
    assert kept == (2,)


def test_plan_calboard_all_is_two_parts():
    channel_sets, kept = allan_plot.plan_capture(args(use_calboard=True))
    assert channel_sets == [(1, 3), (2, 4)]
    assert kept == (0, 1, 2, 3)


def test_plan_calboard_subset_is_one_part():
    channel_sets, kept = allan_plot.plan_capture(
        args(use_calboard=True, calboard_channels=(0, 2))
    )
    assert channel_sets == [(1, 3)]
    assert kept == (0, 2)


def test_plan_calboard_same_bridge_rejected():
    with pytest.raises(allan_plot.AllanError):
        allan_plot.plan_capture(args(use_calboard=True, calboard_channels=(0, 1)))


def test_plan_flag_combinations_rejected():
    with pytest.raises(allan_plot.AllanError):
        allan_plot.plan_capture(args(calboard_channels=(0, 2)))
    with pytest.raises(allan_plot.AllanError):
        allan_plot.plan_capture(args(use_calboard=True, channels=(0,)))


def make_block(ssn0, rows):
    """A raw-unit Block from row tuples; NaN entries mark dropped samples."""
    raw = np.array(rows, dtype=np.float64)
    n = raw.shape[0]
    return Block(
        data=raw.copy(),
        raw=raw,
        t=np.arange(n) / 1000.0,
        ssn0=ssn0,
        units="raw",
        host_time=0.0,
    )


def test_recorder_windows_and_missing(tmp_path):
    rec = allan_plot.AllanRecorder(
        tmp_path / "cap.csv", keep_channels=(0, 2), capacity=16, device_dict={}
    )
    rec.add_block(make_block(100, [(1, 9, 5, 9), (2, 9, 6, 9)]))
    # Two dropped samples occupy ssn 102/103, then the next received sample.
    rec.add_block(make_block(102, [(np.nan,) * 4, (np.nan,) * 4, (3, 9, 7, 9)]))
    assert rec.n_samples == 3
    assert rec.missing_count == 2
    np.testing.assert_array_equal(rec.window(0, 0, 3), [1, 2, 3])
    np.testing.assert_array_equal(rec.window(2, 1, 3), [6, 7])
    rec.cleanup()


def test_recorder_overflow_flags_instead_of_growing(tmp_path):
    rec = allan_plot.AllanRecorder(
        tmp_path / "cap.csv", keep_channels=(0,), capacity=2, device_dict={}
    )
    rec.add_block(make_block(0, [(1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0)]))
    assert rec.overflowed


def test_capture_for_aborts_on_gap():
    rec = SimpleNamespace(overflowed=False, missing_count=3, n_samples=0)
    with pytest.raises(allan_plot.AllanError, match="dropped"):
        asyncio.run(allan_plot.capture_for(rec, 30.0, "x"))


def test_capture_for_aborts_on_overflow():
    rec = SimpleNamespace(overflowed=True, missing_count=0, n_samples=0)
    with pytest.raises(allan_plot.AllanError, match="provisioned"):
        asyncio.run(allan_plot.capture_for(rec, 30.0, "x"))


def test_capture_for_completes_clean_run():
    rec = SimpleNamespace(overflowed=False, missing_count=0, n_samples=0)
    asyncio.run(allan_plot.capture_for(rec, 0.05, "x"))  # returns, no raise


def test_estimate_capacity():
    # regression: 2 parts x 60 s at 1 kSPS with 2 s guards overflowed the old
    # windows-only 1.02 estimate (125k actual rows vs 122k provisioned)
    cap = allan_plot.estimate_capacity(1000, 60.0, 2, 2.0)
    assert cap == 1000 * (60 * 2 + 2 * 2 + 10) + 64
    assert cap >= 60020 + 60000 + 2 * 2000 + 1500
    assert allan_plot.estimate_capacity(1000, 60.0, 1, 0.0) == 1000 * (60 + 10) + 64


def test_hms():
    assert allan_plot._hms(5.9) == "0:00:05"
    assert allan_plot._hms(3661) == "1:01:01"


def test_stdout_tee(tmp_path):
    import io

    console = io.StringIO()
    tee = allan_plot.StdoutTee(console, tmp_path / "r.txt")
    print("feed running", file=tee)
    tee.write("\rprogress 12%  ")
    print("done: 60,020 samples", file=tee)
    print("valley 7.3±0.6 nV at τ=0.431 s — unicode line", file=tee)
    tee.close()
    report = (tmp_path / "r.txt").read_text(encoding="utf-8")
    assert console.getvalue().count("progress") == 1  # console saw the live line
    assert "progress" not in report  # the report did not
    assert "feed running" in report and "done: 60,020 samples" in report
    assert "τ=0.431" in report  # unicode survives into the report
    assert tee.isatty() == console.isatty()  # capture_for asks isatty()


def test_reduce_excludes_railed_channel(tmp_path):
    rec = allan_plot.AllanRecorder(
        tmp_path / "cap.csv", keep_channels=(0, 1), capacity=512, device_dict={}
    )
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 50, 512).astype(int)
    rail = np.full(512, -(1 << 23), dtype=int)
    rec.add_block(make_block(0, list(zip(noise, rail, rail, rail))))
    windows = [{"i0": 0, "i1": rec.n_samples, "dut_channels": (0, 1), "temp_c": None}]
    vpc = [allan_plot.allan_math.volts_per_count(1.2, 101.0, 1)] * 4
    traces = allan_plot.reduce_traces(rec, windows, 1000, vpc)
    assert [t.dut_ch for t in traces] == [0]
    assert traces[0].adev_nv[0] > 0
    assert traces[0].spec_hz.size == traces[0].spec_nv.size > 10
    assert traces[0].asd_hz.size == traces[0].asd_nv.size > 10
    rec.cleanup()


def test_make_plots_writes_three_pngs(tmp_path):
    rec = allan_plot.AllanRecorder(
        tmp_path / "cap.csv", keep_channels=(0, 1), capacity=2048, device_dict={}
    )
    rng = np.random.default_rng(0)
    t = np.arange(2048) / 1000.0
    ch0 = (rng.normal(0, 50, 2048) + 40.0 * np.sin(2 * np.pi * 53.333 * t)).astype(int)
    ch1 = rng.normal(0, 50, 2048).astype(int)
    rec.add_block(make_block(0, list(zip(ch0, ch1, ch1, ch1))))
    windows = [{"i0": 0, "i1": rec.n_samples, "dut_channels": (0, 1), "temp_c": None}]
    vpc = [allan_plot.allan_math.volts_per_count(1.2, 101.0, 1)] * 4
    traces = allan_plot.reduce_traces(rec, windows, 1000, vpc)
    png = tmp_path / "allan.png"
    paths = allan_plot.make_plots(traces, ["Allan deviation — test", "meta"], png, True)
    assert paths == (
        png,
        tmp_path / "allan_spectrum.png",
        tmp_path / "allan_psd.png",
    )
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    rec.cleanup()
