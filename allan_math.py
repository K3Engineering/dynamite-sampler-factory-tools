"""Allan deviation math for ADC noise captures.

Helpers (pure functions and data). Input is a uniformly
sampled, gap-free count series (the math assumes contiguity).

Overlapping Allan deviation via prefix sums: for cluster size m (samples)
over N samples x with S = [0, cumsum(x)],

    OAVAR(m) = sum_i (S[i+2m] - 2*S[i+m] + S[i])^2 / (2 * (N - 2m + 1) * m^2)
    ADEV(m)  = sqrt(OAVAR(m)),   tau = m / sample_rate

in the units of x (counts here; scale the result for volts).
"""

import math

import numpy as np

ADC_COUNTS_PER_POLARITY = 1 << 23  # 24-bit bipolar

# A rail hugging full scale is an unplugged/failed input, not a noise reading.
RAIL_FRACTION = 0.99

POINTS_PER_OCTAVE = 8  # tau grid density; resolves the ~0.2 s valley
MAX_TAU_FRACTION = 8  # largest cluster is N/8: ~8 independent clusters beyond it


def volts_per_count(adc_fsr_v, afe_gain, pga_gain):
    """Volts at the AFE input per ADC count (mirror of cal_math's chain)."""
    return adc_fsr_v / (ADC_COUNTS_PER_POLARITY * afe_gain * pga_gain)


def cluster_sizes(n_samples, points_per_octave=POINTS_PER_OCTAVE):
    """Dense log grid of cluster sizes m (samples), 1 <= m <= n_samples/8."""
    m_max = max(1, n_samples // MAX_TAU_FRACTION)
    sizes = set()
    k = 0
    while (m := round(2 ** (k / points_per_octave))) <= m_max:
        sizes.add(m)
        k += 1
    return sorted(sizes)


def overlapping_adev(x, sample_rate, points_per_octave=POINTS_PER_OCTAVE):
    """(taus_s, adev, err) for count series x sampled at sample_rate.

    err is the 1-sigma relative uncertainty adev/sqrt(K) over K = N/m - 1
    non-overlapped cluster pairs — conservative on purpose (overlapping buys
    extra degrees of freedom, so the bars err wide).
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = x.size
    S = np.empty(n + 1, dtype=np.float64)
    S[0] = 0.0
    np.cumsum(x, out=S[1:])
    taus, adevs, errs = [], [], []
    for m in cluster_sizes(n, points_per_octave):
        D = S[2 * m :] - 2.0 * S[m : n - m + 1] + S[: n - 2 * m + 1]
        oavar = float(D @ D) / (2.0 * (n - 2 * m + 1) * m * m)
        adev = math.sqrt(oavar)
        taus.append(m / sample_rate)
        adevs.append(adev)
        errs.append(adev / math.sqrt(max(1, n // m - 1)))
    return np.array(taus), np.array(adevs), np.array(errs)
