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


def amplitude_spectrum(x, sample_rate):
    """Hann-windowed single-sided peak-amplitude spectrum (freqs_hz, amp).

    Amplitude is in the units of x (counts). Same scaling as detect_peaks.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) * (4.0 / n)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    return freqs, spec


def _welch_nperseg(n):
    cap = min(n, 4096)
    floor = min(n, 256)
    target = min(cap, max(floor, n // 8))
    return min(1 << int(round(math.log2(max(target, 8)))), n)


def noise_asd(x, sample_rate, nperseg=None):
    """One-sided amplitude spectral density via Welch (Hann, 50% overlap).

    Returns (freqs_hz, asd) in units of x / sqrt(Hz).
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if nperseg is None:
        nperseg = _welch_nperseg(n)
    hop = max(1, nperseg // 2)
    w = np.hanning(nperseg)
    w_ss = float(w @ w)
    acc = None
    nseg = 0
    for start in range(0, n - nperseg + 1, hop):
        X = np.fft.rfft(x[start : start + nperseg] * w)
        p = (X.real**2 + X.imag**2) / (sample_rate * w_ss)
        acc = p if acc is None else acc + p
        nseg += 1
    acc /= nseg
    acc[1:] *= 2.0
    if nperseg % 2 == 0:
        acc[-1] *= 0.5
    return np.fft.rfftfreq(nperseg, 1.0 / sample_rate), np.sqrt(acc)


def detect_peaks(x, sample_rate, ratio=6.0, f_min=3.0, max_peaks=12):
    """Narrowband peaks in a count series [(freq_hz, amplitude)].

    Hann-windowed FFT; a peak is above `ratio` x the median bin
    amplitude (greedy max-pick, +-4 bins blanked for the Hann mainlobe).
    Amplitudes are peak counts. Bins below f_min are excluded — that content
    is drift/flicker, not a tone.
    """
    return pick_peaks(*amplitude_spectrum(x, sample_rate), ratio, f_min, max_peaks)


def pick_peaks(freqs, spec, ratio=6.0, f_min=3.0, max_peaks=12):
    """Peak-pick a precomputed amplitude spectrum. See detect_peaks."""
    work = spec.copy()
    work[freqs < f_min] = 0.0
    floor = np.median(spec[freqs >= f_min])
    peaks = []
    for _ in range(max_peaks):
        k = int(np.argmax(work))
        if work[k] <= ratio * floor or work[k] == 0.0:
            break
        peaks.append((float(freqs[k]), float(spec[k])))
        work[max(0, k - 4) : k + 5] = 0.0
    return sorted(peaks)


def format_peaks(peaks, nv_per_count, prefix):
    """One-line report summary of the detected narrowband peaks."""
    if not peaks:
        return f"{prefix}: no narrowband peaks detected"
    parts = [
        f"{f:.1f} Hz: {a:.0f} counts ({a * nv_per_count:.0f} nV)" for f, a in peaks
    ]
    return f"{prefix}: narrowband peaks: " + "; ".join(parts)


def peak_adev(freq, amp, taus_s):
    """ADEV of a sinusoid of amplitude `amp` at `freq`: A sin^2(pi f t)/(pi f t).
    Nulls at tau = k/f — a full-period average of a tone is zero by symmetry."""
    x = np.pi * freq * np.asarray(taus_s, dtype=np.float64)
    return amp * np.sin(x) ** 2 / np.maximum(x, 1e-300)


def narrowband_adev(peaks, taus_s):
    """Quadrature-summed ADEV contribution of all detected peaks."""
    total = np.zeros(len(taus_s))
    for f, a in peaks:
        total += peak_adev(f, a, taus_s) ** 2
    return np.sqrt(total)
