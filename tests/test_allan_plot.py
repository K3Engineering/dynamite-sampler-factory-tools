from types import SimpleNamespace

import numpy as np
import pytest

import allan_plot


def args(**kw):
    base = {
        "use_calboard": False,
        "channels": None,
        "calboard_channels": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_plan_passive_defaults_to_all_channels():
    phases, kept = allan_plot.plan_capture(args())
    assert phases is None
    assert kept == (0, 1, 2, 3)


def test_plan_passive_channel_subset():
    phases, kept = allan_plot.plan_capture(args(channels=(2,)))
    assert phases is None
    assert kept == (2,)


def test_plan_calboard_all_is_two_phases():
    phases, kept = allan_plot.plan_capture(args(use_calboard=True))
    assert phases == [(1, 3), (2, 4)]
    assert kept == (0, 1, 2, 3)


def test_plan_calboard_subset_is_one_phase():
    phases, kept = allan_plot.plan_capture(
        args(use_calboard=True, calboard_channels=(0, 2))
    )
    assert phases == [(1, 3)]
    assert kept == (0, 2)


def test_plan_calboard_same_bridge_rejected():
    with pytest.raises(allan_plot.AllanError):
        allan_plot.plan_capture(args(use_calboard=True, calboard_channels=(0, 1)))


def test_plan_flag_combinations_rejected():
    with pytest.raises(allan_plot.AllanError):
        allan_plot.plan_capture(args(calboard_channels=(0, 2)))
    with pytest.raises(allan_plot.AllanError):
        allan_plot.plan_capture(args(use_calboard=True, channels=(0,)))


def make_packet(base_ssn, counts):
    header = SimpleNamespace(sample_sequence_number=base_ssn)
    # the recorder only needs attribute access, so namespaces stand in for FeedData
    return header, [
        SimpleNamespace(ch0=c[0], ch1=c[1], ch2=c[2], ch3=c[3]) for c in counts
    ]


def test_recorder_windows_and_missing(tmp_path):
    rec = allan_plot.AllanRecorder(
        tmp_path / "cap.csv", keep_channels=(0, 2), capacity=16
    )
    rec.setup({})
    header, feed = make_packet(100, [(1, 9, 5, 9), (2, 9, 6, 9)])
    rec.callback(header, feed, 0)
    header, feed = make_packet(102, [(3, 9, 7, 9)])
    rec.callback(header, feed, 2)  # 2 dropped samples reported
    assert rec.n_samples == 3
    assert rec.missing_count == 2
    np.testing.assert_array_equal(rec.window(0, 0, 3), [1, 2, 3])
    np.testing.assert_array_equal(rec.window(2, 1, 3), [6, 7])
    rec.cleanup()


def test_recorder_overflow_flags_instead_of_growing(tmp_path):
    rec = allan_plot.AllanRecorder(tmp_path / "cap.csv", keep_channels=(0,), capacity=2)
    rec.setup({})
    header, feed = make_packet(0, [(1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0)])
    rec.callback(header, feed, 0)
    assert rec.overflowed


def test_reduce_excludes_railed_channel(tmp_path):
    rec = allan_plot.AllanRecorder(
        tmp_path / "cap.csv", keep_channels=(0, 1), capacity=512
    )
    rec.setup({})
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 50, 512).astype(int)
    rail = np.full(512, -(1 << 23), dtype=int)
    header, feed = make_packet(0, list(zip(noise, rail, rail, rail)))
    rec.callback(header, feed, 0)
    windows = [{"i0": 0, "i1": rec.n_samples, "dut_channels": (0, 1), "temp_c": None}]
    vpc = [allan_plot.allan_math.volts_per_count(1.2, 101.0, 1)] * 4
    traces = allan_plot.reduce_traces(rec, windows, 1000, vpc)
    assert [t.dut_ch for t in traces] == [0]
    assert traces[0].adev_nv[0] > 0
    rec.cleanup()
