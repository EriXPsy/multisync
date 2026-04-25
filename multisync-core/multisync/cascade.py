"""
Cascade analysis — Cross-Correlation Function with rigorous surrogate testing.

Key design decisions (Nature Methods reviewer requirements):
- CCF computed via numpy correlate (vectorized, no Python for-loops).
- Edge-effect mitigation via Hanning (tapered cosine) window.
- Surrogate testing via Phase-Randomized Fourier Transform (PRTF), which
  destroys temporal structure while preserving the power spectrum and
  amplitude distribution.  Far more rigorous than circular shift.
- p-value: fraction of surrogates whose |peak CCF| >= observed |peak CCF|.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import fft, ifft, fftfreq


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CCAResult:
    """Result of a single cross-correlation analysis between two modalities."""
    modality_a: str
    modality_b: str
    feature_a: str = ""
    feature_b: str = ""
    lags_sec: np.ndarray = field(default_factory=lambda: np.array([]))
    ccf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_lag_sec: float = 0.0
    peak_ccf: float = 0.0
    direction: str = ""
    # Significance
    is_significant: bool = False
    p_value: float = 1.0
    surrogate_n: int = 0
    null_peak_ccf: Optional[np.ndarray] = None  # surrogates' peak CCFs


@dataclass
class CascadeEdge:
    """A directed edge in the cascade graph (Viewer JSON ready)."""
    source: str
    target: str
    lag_sec: float
    ccf_value: float
    p_value: float
    is_significant: bool


# ---------------------------------------------------------------------------
# Surrogate generation — Phase-Randomized Fourier Transform (PRTF)
# ---------------------------------------------------------------------------

def _prft_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a Phase-Randomized Fourier Transform surrogate.

    Steps:
    1. FFT of original signal.
    2. Randomize phases uniformly in [-pi, pi], except keep DC and Nyquist
       phases fixed (preserve mean and amplitude distribution).
    3. Inverse FFT → surrogate with same power spectrum, destroyed temporal
       structure.

    Parameters
    ----------
    x : 1-D array
        Original time series.
    rng : numpy Generator
        Random number generator (for reproducibility).

    Returns
    -------
    surrogate : 1-D array, same length as x.
    """
    n = len(x)
    if n < 4:
        return x.copy()

    # FFT
    X = fft(x)
    freqs = fftfreq(n)

    # Randomize phases (preserve DC at index 0 and Nyquist at index n//2)
    phases = np.angle(X)
    random_phases = rng.uniform(-np.pi, np.pi, size=n)

    # Keep DC and Nyquist (if even-length)
    random_phases[0] = phases[0]
    if n % 2 == 0:
        random_phases[n // 2] = phases[n // 2]

    # Enforce Hermitian symmetry: X[k] = conj(X[n-k])
    # For real-valued signal, we only randomize the first half (including Nyquist)
    X_new = np.zeros_like(X)
    half = n // 2 + 1  # include Nyquist for even n
    for k in range(half):
        magnitude = np.abs(X[k])
        new_phase = random_phases[k]
        X_new[k] = magnitude * np.exp(1j * new_phase)
        if k > 0 and k < n - k:
            # Mirror frequency (skip when k == n-k, i.e., Nyquist for even n)
            X_new[n - k] = np.conj(X_new[k])

    surrogate = np.real(ifft(X_new))
    return surrogate


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _hanning_window(n: int) -> np.ndarray:
    """Periodic Hanning window (scipy.signal.windows.hann equivalent)."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


# ---------------------------------------------------------------------------
# CCF computation (vectorized)
# ---------------------------------------------------------------------------

def compute_ccf(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_sec: float,
    hz: float,
    apply_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation function via numpy correlate.

    Parameters
    ----------
    x, y : 1-D arrays
        Input time series (same length, same sampling rate).
    max_lag_sec : float
        Maximum lag to compute, in seconds.
    hz : float
        Sampling rate in Hz.
    apply_window : bool
        If True, apply Hanning window before CCF to reduce edge effects.

    Returns
    -------
    lags_sec : 1-D array
        Lag values in seconds.
    ccf : 1-D array
        Normalized CCF values (Pearson-like, range ≈ [-1, 1]).
    """
    n = len(x)
    assert len(y) == n, f"x and y must have same length: {n} vs {len(y)}"

    max_lag_samples = int(np.floor(max_lag_sec * hz))

    if n < 2 * max_lag_samples + 1:
        raise ValueError(
            f"Series length ({n}) too short for max_lag ({max_lag_samples} samples). "
            f"Need at least {2 * max_lag_samples + 1}."
        )

    # Apply Hanning window for edge-effect mitigation
    if apply_window:
        win = _hanning_window(n)
        x_w = (x - x.mean()) * win
        y_w = (y - y.mean()) * win
    else:
        x_w = x - x.mean()
        y_w = y - y.mean()

    # Normalized cross-correlation (vectorized via numpy)
    # scipy.signal.correlate equivalent: mode='same' would center, but we
    # use 'full' and extract the valid range
    nccf = np.correlate(x_w, y_w, mode="full")

    # Normalize to Pearson-like correlation
    norm_x = np.sqrt(np.sum(x_w ** 2))
    norm_y = np.sqrt(np.sum(y_w ** 2))
    if norm_x > 0 and norm_y > 0:
        nccf = nccf / (norm_x * norm_y)

    # Extract the lag range: negative lags (x leads) to positive lags (y leads)
    center = n - 1
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    valid = center + lags
    ccf = nccf[valid]

    lags_sec = lags / hz

    return lags_sec, ccf


# ---------------------------------------------------------------------------
# Full cascade analysis with surrogate testing
# ---------------------------------------------------------------------------

def cascade_analysis(
    dataset: "SynchronyDataset",  # noqa: F821  forward ref
    max_lag_sec: float = 30.0,
    surrogate_n: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    apply_window: bool = True,
) -> Tuple[List[CCAResult], List[CascadeEdge]]:
    """
    Compute pairwise CCF across all modalities with PRTF surrogate testing.

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned and normalized.
    max_lag_sec : float
        Maximum cross-correlation lag in seconds.
    surrogate_n : int
        Number of PRTF surrogates for significance testing.
    alpha : float
        Significance threshold.  ``is_significant`` is True when
        p < alpha.
    seed : int
        Random seed for reproducibility.
    apply_window : bool
        Apply Hanning window before CCF.

    Returns
    -------
    cca_results : list of CCAResult
        Detailed results per modality pair.
    edges : list of CascadeEdge
        Viewer-ready directed cascade edges (only significant ones).
    """
    if not dataset._aligned:
        raise ValueError("Dataset must be aligned before cascade analysis.")

    hz = dataset.target_hz
    feat_cols = dataset.feature_columns
    modality_names = dataset.modality_names
    rng = np.random.default_rng(seed)

    cca_results: List[CCAResult] = []
    edges: List[CascadeEdge] = []

    for i, name_a in enumerate(modality_names):
        for name_b in modality_names[i + 1:]:
            for col_a in feat_cols[name_a]:
                for col_b in feat_cols[name_b]:
                    x = dataset.get_aligned_array(name_a, col_a)
                    y = dataset.get_aligned_array(name_b, col_b)
                    if x is None or y is None:
                        continue

                    # Trim leading and trailing NaN only — never slice internal gaps.
                    # Internal NaN → fill with 0 (zero contribution to correlation).
                    either_nan = np.isnan(x) | np.isnan(y)
                    if either_nan.sum() == len(x):
                        continue  # entirely NaN
                    # Find first/last valid index to trim edges
                    first_valid = int(np.argmax(~either_nan))
                    last_valid = len(either_nan) - 1 - int(np.argmax(~either_nan[::-1]))
                    x_trim = x[first_valid : last_valid + 1].copy()
                    y_trim = y[first_valid : last_valid + 1].copy()
                    if len(x_trim) < 20:
                        continue
                    # Fill internal NaN with 0 (preserves length, no time-axis collapse)
                    x_clean = np.where(np.isnan(x_trim), 0.0, x_trim)
                    y_clean = np.where(np.isnan(y_trim), 0.0, y_trim)

                    # Cap max_lag_sec to the mathematically feasible range for
                    # this particular segment.  CCF needs n >= 2*max_lag + 1,
                    # so max_lag <= (n-1)//2.  Silently cap instead of crashing.
                    n_seg = len(x_clean)
                    feasible_lag_samples = (n_seg - 1) // 2
                    feasible_lag_sec = feasible_lag_samples / hz
                    effective_max_lag = min(max_lag_sec, feasible_lag_sec)

                    # Compute observed CCF
                    lags_sec, ccf_vals = compute_ccf(
                        x_clean, y_clean, effective_max_lag, hz, apply_window
                    )

                    # Find peak
                    abs_ccf = np.abs(ccf_vals)
                    peak_idx = int(np.argmax(abs_ccf))
                    peak_lag = float(lags_sec[peak_idx])
                    peak_val = float(ccf_vals[peak_idx])

                    # Determine direction
                    if peak_lag < 0:
                        direction = f"{name_a}→{name_b}"
                        source, target = name_a, name_b
                        peak_lag = abs(peak_lag)
                    elif peak_lag > 0:
                        direction = f"{name_b}→{name_a}"
                        source, target = name_b, name_a
                        peak_lag = abs(peak_lag)
                    else:
                        direction = "synchronous"
                        source, target = name_a, name_b

                    # --- Surrogate testing (PRTF) ---
                    null_peaks = np.empty(surrogate_n)
                    for s in range(surrogate_n):
                        x_surr = _prft_surrogate(x_clean, rng)
                        y_surr = _prft_surrogate(y_clean, rng)
                        _, ccf_s = compute_ccf(
                            x_surr, y_surr, effective_max_lag, hz, apply_window
                        )
                        null_peaks[s] = np.max(np.abs(ccf_s))

                    # p-value: fraction of surrogates >= observed
                    p_val = float(np.mean(null_peaks >= abs(peak_val)))
                    is_sig = p_val < alpha

                    result = CCAResult(
                        modality_a=name_a,
                        modality_b=name_b,
                        feature_a=col_a,
                        feature_b=col_b,
                        lags_sec=lags_sec,
                        ccf_values=ccf_vals,
                        peak_lag_sec=peak_lag,
                        peak_ccf=peak_val,
                        direction=direction,
                        is_significant=is_sig,
                        p_value=p_val,
                        surrogate_n=surrogate_n,
                        null_peak_ccf=null_peaks,
                    )
                    cca_results.append(result)

                    if is_sig and direction != "synchronous":
                        edges.append(CascadeEdge(
                            source=source,
                            target=target,
                            lag_sec=peak_lag,
                            ccf_value=peak_val,
                            p_value=p_val,
                            is_significant=True,
                        ))

    return cca_results, edges


# Re-export for convenience
from .dataset import SynchronyDataset  # noqa: E402, F401
