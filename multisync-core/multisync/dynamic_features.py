"""
Dynamic feature extraction — Gordon-inspired synchrony dynamics.

10 features organized by temporal phase:
  Onset:    onset_latency, onset_amplitude
  Build-up: build_up_rate, build_up_slope
  Maintenance: peak_amplitude, peak_duration
  Breakdown: breakdown_rate, recovery_time
  Global:   mean_synchrony, synchrony_entropy

All computations are vectorized (numpy/scipy).  Edge effects handled
via Hanning window where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy as sp_entropy


# ---------------------------------------------------------------------------
# Sliding-window WCC (Weighted Cross-Correlation)
# ---------------------------------------------------------------------------

def sliding_window_wcc(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    lag_samples: int = 0,
) -> np.ndarray:
    """
    Compute sliding-window cross-correlation (WCC) between x and y.

    For each window position, computes Pearson correlation within the window.
    Vectorized implementation using stride tricks.

    Parameters
    ----------
    x, y : 1-D arrays
        Input time series (same length).
    window_size : int
        Window size in samples.
    hz : float
        Sampling rate (for time axis, not used in computation directly).
    lag_samples : int
        Lag y by this many samples before correlating.

    Returns
    -------
    wcc : 1-D array
        Cross-correlation at each window position. Length = len(x) - window_size + 1.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError(f"x and y must have same length: {n} vs {len(y)}")
    if window_size > n:
        # Graceful fallback: return empty array.  Downstream feature
        # extraction already handles empty arrays (returns NaN features).
        return np.array([], dtype=float)

    # Apply lag
    if lag_samples > 0:
        y_lagged = np.full(n, np.nan)
        y_lagged[lag_samples:] = y[:-lag_samples]
    elif lag_samples < 0:
        y_lagged = np.full(n, np.nan)
        y_lagged[:lag_samples] = y[-lag_samples:]
    else:
        y_lagged = y

    # Use stride tricks for vectorized windowing
    from numpy.lib.stride_tricks import sliding_window_view

    x_windows = sliding_window_view(x, window_size)  # shape (n-w+1, w)
    y_windows = sliding_window_view(y_lagged, window_size)

    # Pearson correlation per window
    x_means = x_windows.mean(axis=1, keepdims=True)
    y_means = y_windows.mean(axis=1, keepdims=True)
    x_centered = x_windows - x_means
    y_centered = y_windows - y_means

    x_std = np.sqrt((x_centered ** 2).sum(axis=1))
    y_std = np.sqrt((y_centered ** 2).sum(axis=1))

    # Avoid division by zero
    denom = x_std * y_std
    denom[denom == 0] = 1.0

    wcc = (x_centered * y_centered).sum(axis=1) / denom
    wcc = np.clip(wcc, -1.0, 1.0)

    # Mask windows with any NaN
    nan_mask = np.isnan(x_windows).any(axis=1) | np.isnan(y_windows).any(axis=1)
    wcc[nan_mask] = np.nan

    return wcc


# ---------------------------------------------------------------------------
# Feature data class
# ---------------------------------------------------------------------------

@dataclass
class DynamicFeatures:
    """Container for 10 Gordon-inspired dynamic features."""
    # Onset
    onset_latency: float       # seconds from start to first significant sync
    onset_amplitude: float     # WCC value at onset point
    # Build-up
    build_up_rate: float       # (peak - onset) / (peak_time - onset_time)
    build_up_slope: float      # linear slope of WCC during build-up (sec^-1)
    # Maintenance
    peak_amplitude: float      # maximum WCC value
    peak_duration: float       # seconds WCC stays above 75% of peak
    # Breakdown
    breakdown_rate: float      # rate of decrease after peak (per second)
    recovery_time: float       # seconds from peak to WCC dropping below onset level
    # Global
    mean_synchrony: float      # mean WCC over entire recording
    synchrony_entropy: float   # Shannon entropy of WCC distribution

    def to_dict(self) -> Dict[str, float]:
        return {
            "onset_latency": self.onset_latency,
            "onset_amplitude": self.onset_amplitude,
            "build_up_rate": self.build_up_rate,
            "build_up_slope": self.build_up_slope,
            "peak_amplitude": self.peak_amplitude,
            "peak_duration": self.peak_duration,
            "breakdown_rate": self.breakdown_rate,
            "recovery_time": self.recovery_time,
            "mean_synchrony": self.mean_synchrony,
            "synchrony_entropy": self.synchrony_entropy,
        }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_dynamic_features(
    wcc: np.ndarray,
    hz: float = 1.0,
    onset_threshold: float = 0.2,
) -> DynamicFeatures:
    """
    Extract 10 dynamic features from a WCC time series.

    Parameters
    ----------
    wcc : 1-D array
        Windowed cross-correlation time series.
    hz : float
        Sampling rate of WCC (Hz).
    onset_threshold : float
        Minimum WCC value to count as "significant synchrony onset".

    Returns
    -------
    DynamicFeatures with all 10 features.
    """
    # Handle NaN
    valid = ~np.isnan(wcc)
    if valid.sum() < 5:
        empty = DynamicFeatures(
            onset_latency=np.nan, onset_amplitude=np.nan,
            build_up_rate=np.nan, build_up_slope=np.nan,
            peak_amplitude=np.nan, peak_duration=np.nan,
            breakdown_rate=np.nan, recovery_time=np.nan,
            mean_synchrony=np.nan, synchrony_entropy=np.nan,
        )
        return empty

    wcc_clean = wcc.copy()
    wcc_clean[~valid] = 0.0
    dt = 1.0 / hz

    # --- Onset ---
    onset_indices = np.where(wcc_clean >= onset_threshold)[0]
    if len(onset_indices) > 0:
        onset_idx = int(onset_indices[0])
        onset_latency = onset_idx * dt
        onset_amplitude = float(wcc_clean[onset_idx])
    else:
        onset_idx = 0
        onset_latency = np.nan
        onset_amplitude = np.nan

    # --- Peak ---
    # Use -inf to fill NaN positions so they can NEVER be selected as peak.
    # This prevents zero-padded NaN slots from creating false peaks when
    # all valid WCC values are negative.
    wcc_valid_only = wcc.copy()
    wcc_valid_only[~valid] = -np.inf
    peak_idx = int(np.argmax(wcc_valid_only))
    peak_amplitude = float(wcc_valid_only[peak_idx])
    peak_time = peak_idx * dt

    # --- Build-up ---
    if onset_idx < peak_idx and not np.isnan(onset_latency):
        build_up_duration = (peak_idx - onset_idx) * dt
        build_up_rate = (peak_amplitude - onset_amplitude) / build_up_duration if build_up_duration > 0 else 0.0
        # Linear slope during build-up window
        if peak_idx - onset_idx > 1:
            build_up_segment = wcc_clean[onset_idx:peak_idx + 1]
            t_segment = np.arange(len(build_up_segment)) * dt
            slope, _ = np.polyfit(t_segment, build_up_segment, 1)
            build_up_slope = float(slope)
        else:
            build_up_slope = 0.0
    else:
        build_up_rate = 0.0
        build_up_slope = 0.0

    # --- Peak duration (time above 75% of peak) ---
    if peak_amplitude > 0:
        threshold_75 = 0.75 * peak_amplitude
        above_75 = wcc_clean >= threshold_75
        peak_duration = float(np.sum(above_75) * dt)
    else:
        peak_duration = 0.0

    # --- Breakdown ---
    if peak_idx < len(wcc_clean) - 1:
        post_peak = wcc_clean[peak_idx:]
        # Rate of decrease: linear slope after peak
        if len(post_peak) > 1:
            t_post = np.arange(len(post_peak)) * dt
            slope, _ = np.polyfit(t_post, post_peak, 1)
            breakdown_rate = -float(slope)  # positive = decreasing
        else:
            breakdown_rate = 0.0

        # Recovery time: time from peak to WCC dropping below onset level
        if onset_amplitude is not np.nan and not np.isnan(onset_amplitude):
            recovery_mask = post_peak < onset_amplitude
            if recovery_mask.any():
                recovery_samples = np.argmax(recovery_mask)
                recovery_time = recovery_samples * dt
            else:
                recovery_time = float((len(wcc_clean) - 1 - peak_idx) * dt)
        else:
            recovery_time = np.nan
    else:
        breakdown_rate = 0.0
        recovery_time = np.nan

    # --- Global features ---
    mean_synchrony = float(np.mean(wcc_clean[valid]))

    # Shannon entropy of binned WCC distribution
    wcc_valid = wcc_clean[valid]
    if len(wcc_valid) > 1 and np.std(wcc_valid) > 0:
        hist, _ = np.histogram(wcc_valid, bins=10, density=True)
        hist = hist[hist > 0]  # remove zero bins
        hist = hist / hist.sum()  # normalize
        synchrony_entropy = float(sp_entropy(hist))
    else:
        synchrony_entropy = 0.0

    return DynamicFeatures(
        onset_latency=onset_latency,
        onset_amplitude=onset_amplitude,
        build_up_rate=build_up_rate,
        build_up_slope=build_up_slope,
        peak_amplitude=peak_amplitude,
        peak_duration=peak_duration,
        breakdown_rate=breakdown_rate,
        recovery_time=recovery_time,
        mean_synchrony=mean_synchrony,
        synchrony_entropy=synchrony_entropy,
    )


def extract_features_all_pairs(
    dataset: "SynchronyDataset",  # noqa: F821
    window_size: int = 10,
    hz: float = 1.0,
    onset_threshold: float = 0.2,
) -> Dict[str, DynamicFeatures]:
    """
    Compute WCC + dynamic features for all modality pairs.

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned and normalized.
    window_size : int
        WCC window size in samples.
    hz : float
        Sampling rate.
    onset_threshold : float
        WCC threshold for onset detection.

    Returns
    -------
    dict mapping "modA_modB" → DynamicFeatures.
    """
    feat_cols = dataset.feature_columns
    names = dataset.modality_names
    results: Dict[str, DynamicFeatures] = {}

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            for col_a in feat_cols[name_a]:
                for col_b in feat_cols[name_b]:
                    x = dataset.get_aligned_array(name_a, col_a)
                    y = dataset.get_aligned_array(name_b, col_b)
                    if x is None or y is None:
                        continue

                    wcc = sliding_window_wcc(x, y, window_size, hz)
                    feat = extract_dynamic_features(wcc, hz, onset_threshold)
                    key = f"{name_a}_{col_a}__{name_b}_{col_b}"
                    results[key] = feat

    return results
