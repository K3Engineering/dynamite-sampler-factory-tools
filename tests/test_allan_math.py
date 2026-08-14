import math

import numpy as np

import allan_math

SEEDS = range(8)


def white_ratios(m, n=32768, sigma=13.0):
    """ADEV/sigma/sqrt(m) over several seeds — one seed scatters too much."""
    out = []
    for seed in SEEDS:
        x = np.random.default_rng(seed).normal(0.0, sigma, n)
        taus, adev, _ = allan_math.overlapping_adev(x, 1000.0)
        i = list(taus * 1000.0).index(m)  # fs=1 kHz -> tau in ms = m
        out.append(adev[i] / (sigma / math.sqrt(m)))
    return out


def test_white_noise_tracks_sqrt_m():
    """ADEV of white gaussian noise with per-sample std sigma is sigma/sqrt(m)."""
    for m in (16, 256, 1024):
        ratios = white_ratios(m)
        assert abs(np.mean(ratios) - 1.0) < 0.08, f"m={m}: mean {np.mean(ratios)}"
        assert all(0.7 < r < 1.3 for r in ratios), f"m={m}: {ratios}"


def test_exact_small_sequence():
    """Alternating [0,1,...]: all +-1 first differences, identical m=2 clusters."""
    x = np.array([0, 1] * 8, dtype=np.float64)  # N=16 -> grid capped at m=2
    taus, adev, _ = allan_math.overlapping_adev(x, 1.0)
    assert list(taus) == [1.0, 2.0]
    assert math.isclose(adev[0], math.sqrt(0.5))
    assert adev[1] == 0.0


def test_constant_input_is_zero():
    _, adev, _ = allan_math.overlapping_adev(np.full(4096, 7), 1000.0)
    assert np.all(adev == 0.0)


def test_cluster_grid():
    sizes = allan_math.cluster_sizes(100_000)
    m_max = 100_000 // allan_math.MAX_TAU_FRACTION
    assert sizes[0] == 1
    assert m_max / 1.2 < sizes[-1] <= m_max  # grid reaches the cap within a step
    assert len(sizes) == len(set(sizes))  # dense round() grid deduplicated
    assert all(isinstance(m, int) and 1 <= m <= m_max for m in sizes)
    assert allan_math.cluster_sizes(4) == [1]


def test_error_bars():
    n = 32768
    errs, one_over_sqrt = [], 1.0 / math.sqrt(n - 1)
    for seed in SEEDS:
        x = np.random.default_rng(seed).normal(0.0, 1.0, n)
        _, _, err = allan_math.overlapping_adev(x, 1000.0)
        errs.append(
            err[0] / one_over_sqrt
        )  # m=1: err = adev/sqrt(n-1) ~ sigma/sqrt(n-1)
        assert np.all(np.isfinite(err)) and np.all(err > 0)
    assert abs(np.mean(errs) - 1.0) < 0.03


def test_volts_per_count():
    v = allan_math.volts_per_count(1.2, 101.0, 1)
    assert abs(v - 1.2 / (2**23 * 101.0)) < 1e-15
    assert 1.4e-9 < v < 1.5e-9  # ~1.42 nV/count on v700P
