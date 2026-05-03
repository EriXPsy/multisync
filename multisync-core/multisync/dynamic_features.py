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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .dataset import SynchronyDataset

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy as sp_entropy
import logging

logger = logging.getLogger(__name__)


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
    Uses cumsum-based O(n) memory implementation when there are no NaN values;
    falls back to stride_tricks (O(n*w) memory) when NaN values are present.

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

    # --- Strategy selection ---
    if np.isnan(x).any() or np.isnan(y_lagged).any():
        # Fallback: stride_tricks (original method)
        # Warn if memory usage would be excessive
        mem_estimate = (n - window_size + 1) * window_size * 8 * 4  # 4 arrays, 8 bytes per float64
        if mem_estimate > 1e9:  # > 1GB
            logger.warning(
                f"sliding_window_wcc: large memory estimate ({mem_estimate/1e9:.1f} GB) "
                f"due to NaN values forcing stride_tricks fallback. "
                f"Consider filling NaN before calling this function."
            )
        return _sliding_window_wcc_stride(x, y_lagged, window_size)
    else:
        # Fast path: cumsum-based O(n) memory
        return _sliding_window_wcc_cumsum(x, y_lagged, window_size)


def _sliding_window_wcc_cumsum(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Cumsum-based sliding-window Pearson correlation.
    Assumes no NaN values in x or y.
    Memory: O(n) instead of O(n*w).
    """
    n = len(x)
    # Pad cumsum with leading 0 for easier window sum computation
    cumsum_x = np.cumsum(x)
    cumsum_y = np.cumsum(y)
    cumsum_xy = np.cumsum(x * y)
    cumsum_x2 = np.cumsum(x ** 2)
    cumsum_y2 = np.cumsum(y ** 2)

    # Pad with 0 at the beginning
    cumsum_x = np.concatenate([[0.0], cumsum_x])
    cumsum_y = np.concatenate([[0.0], cumsum_y])
    cumsum_xy = np.concatenate([[0.0], cumsum_xy])
    cumsum_x2 = np.concatenate([[0.0], cumsum_x2])
    cumsum_y2 = np.concatenate([[0.0], cumsum_y2])

    # Window indices: i = 0 .. n-window_size
    i = np.arange(n - window_size + 1)
    i_end = i + window_size

    sum_x = cumsum_x[i_end] - cumsum_x[i]
    sum_y = cumsum_y[i_end] - cumsum_y[i]
    sum_xy = cumsum_xy[i_end] - cumsum_xy[i]
    sum_x2 = cumsum_x2[i_end] - cumsum_x2[i]
    sum_y2 = cumsum_y2[i_end] - cumsum_y2[i]

    w = float(window_size)
    mean_x = sum_x / w
    mean_y = sum_y / w

    # Variance: E[X^2] - E[X]^2
    var_x = (sum_x2 / w) - mean_x ** 2
    var_y = (sum_y2 / w) - mean_y ** 2

    # Covariance: E[XY] - E[X]E[Y]
    cov = (sum_xy / w) - mean_x * mean_y

    # Pearson r = cov / (std_x * std_y)
    # Handle numerical errors: var could be slightly negative
    var_x = np.maximum(var_x, 0.0)
    var_y = np.maximum(var_y, 0.0)
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    denom = std_x * std_y
    wcc = np.full_like(sum_x, np.nan)
    valid = denom > 1e-10
    wcc[valid] = cov[valid] / denom[valid]

    return np.clip(wcc, -1.0, 1.0)


def _sliding_window_wcc_stride(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Original stride_tricks implementation (O(n*w) memory).
    Handles NaN by masking windows that contain any NaN.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    x_windows = sliding_window_view(x, window_size)  # shape (n-w+1, w)
    y_windows = sliding_window_view(y, window_size)

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

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "DynamicFeatures":
        """Deserialize from a dict (inverse of to_dict)."""
        return cls(
            onset_latency=float(d.get("onset_latency", np.nan)),
            onset_amplitude=float(d.get("onset_amplitude", np.nan)),
            build_up_rate=float(d.get("build_up_rate", np.nan)),
            build_up_slope=float(d.get("build_up_slope", np.nan)),
            peak_amplitude=float(d.get("peak_amplitude", np.nan)),
            peak_duration=float(d.get("peak_duration", np.nan)),
            breakdown_rate=float(d.get("breakdown_rate", np.nan)),
            recovery_time=float(d.get("recovery_time", np.nan)),
            mean_synchrony=float(d.get("mean_synchrony", np.nan)),
            synchrony_entropy=float(d.get("synchrony_entropy", np.nan)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DynamicFeatures":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_dynamic_features(
    wcc: np.ndarray,
    hz: float = 1.0,
    onset_threshold: float = 0.2,
    max_nan_ratio: float = 0.2,
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
    max_nan_ratio : float
        Maximum fraction of NaN values permitted.  If the NaN ratio exceeds
        this threshold, all features are returned as NaN.  Default 0.2 (20%).
        This prevents spurious dynamic features being extracted from segments
        where structural missingness (e.g., face tracking dropout) dominates.

    Returns
    -------
    DynamicFeatures with all 10 features.
    """
    _nan_features = DynamicFeatures(
        onset_latency=np.nan, onset_amplitude=np.nan,
        build_up_rate=np.nan, build_up_slope=np.nan,
        peak_amplitude=np.nan, peak_duration=np.nan,
        breakdown_rate=np.nan, recovery_time=np.nan,
        mean_synchrony=np.nan, synchrony_entropy=np.nan,
    )

    # Hard NaN-ratio guard: structural missingness → reject the whole window
    valid = ~np.isnan(wcc)
    nan_ratio = 1.0 - valid.mean()
    if nan_ratio > max_nan_ratio or valid.sum() < 5:
        return _nan_features

    # Work on a copy where NaN positions stay NaN (never zero-fill here).
    # Zero-filling corrupts onset detection (wcc==0 is "below threshold" but
    # not missing) and distorts linear slope fits for build-up / breakdown.
    wcc_valid_only = wcc.copy()  # NaN positions remain NaN
    dt = 1.0 / hz

    # --- Onset ---
    # Use only valid (non-NaN) positions for threshold detection.
    onset_indices = np.where(valid & (wcc_valid_only >= onset_threshold))[0]
    if len(onset_indices) > 0:
        onset_idx = int(onset_indices[0])
        onset_latency = onset_idx * dt
        onset_amplitude = float(wcc_valid_only[onset_idx])
    else:
        onset_idx = 0
        onset_latency = np.nan
        onset_amplitude = np.nan

    # --- Peak ---
    # Use -inf to fill NaN positions so they can NEVER be selected as peak.
    # This prevents zero-padded NaN slots from creating false peaks when
    # all valid WCC values are negative.
    wcc_for_peak = wcc_valid_only.copy()
    wcc_for_peak[~valid] = -np.inf
    peak_idx = int(np.argmax(wcc_for_peak))
    peak_amplitude = float(wcc_for_peak[peak_idx])
    peak_time = peak_idx * dt

    # --- Build-up ---
    if onset_idx < peak_idx and not np.isnan(onset_latency):
        build_up_duration = (peak_idx - onset_idx) * dt
        build_up_rate = (peak_amplitude - onset_amplitude) / build_up_duration if build_up_duration > 0 else 0.0
        # Linear slope during build-up window — skip NaN positions
        if peak_idx - onset_idx > 1:
            build_up_segment = wcc_valid_only[onset_idx:peak_idx + 1]
            bu_valid = ~np.isnan(build_up_segment)
            if bu_valid.sum() > 1:
                t_segment = np.arange(len(build_up_segment))[bu_valid] * dt
                slope, _ = np.polyfit(t_segment, build_up_segment[bu_valid], 1)
                build_up_slope = float(slope)
            else:
                build_up_slope = 0.0
        else:
            build_up_slope = 0.0
    else:
        build_up_rate = 0.0
        build_up_slope = 0.0

    # --- Peak duration (time above 75% of peak) ---
    if peak_amplitude > 0:
        threshold_75 = 0.75 * peak_amplitude
        # Only count valid (non-NaN) positions above threshold
        above_75 = valid & (wcc_valid_only >= threshold_75)
        peak_duration = float(np.sum(above_75) * dt)
    else:
        peak_duration = 0.0

    # --- Breakdown ---
    if peak_idx < len(wcc_valid_only) - 1:
        post_peak = wcc_valid_only[peak_idx:]
        post_valid = ~np.isnan(post_peak)
        # Rate of decrease: linear slope after peak (valid points only)
        if post_valid.sum() > 1:
            t_post = np.arange(len(post_peak))[post_valid] * dt
            slope, _ = np.polyfit(t_post, post_peak[post_valid], 1)
            breakdown_rate = -float(slope)  # positive = decreasing
        else:
            breakdown_rate = 0.0

        # Recovery time: time from peak to WCC dropping below onset level
        if not np.isnan(onset_amplitude):
            recovery_mask = post_valid & (post_peak < onset_amplitude)
            if recovery_mask.any():
                recovery_samples = np.argmax(recovery_mask)
                recovery_time = recovery_samples * dt
            else:
                recovery_time = float((len(wcc_valid_only) - 1 - peak_idx) * dt)
        else:
            recovery_time = np.nan
    else:
        breakdown_rate = 0.0
        recovery_time = np.nan

    # --- Global features (only over valid samples) ---
    wcc_valid_vals = wcc_valid_only[valid]
    mean_synchrony = float(np.mean(wcc_valid_vals))

    # Shannon entropy of binned WCC distribution
    if len(wcc_valid_vals) > 1 and np.std(wcc_valid_vals) > 0:
        hist, _ = np.histogram(wcc_valid_vals, bins=10, density=True)
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


def extract_features_segmented(
    dataset: "SynchronyDataset",  # noqa: F821
    window_size: int = 10,
    hz: float = 1.0,
    onset_threshold: float = 0.2,
    max_nan_ratio: float = 0.2,
) -> Dict[str, Dict[str, DynamicFeatures]]:
    """
    Compute WCC + dynamic features per CONTEXT segment.

    When a dataset has context labels (e.g., "Rest", "Task", "Cooperation"),
    this function extracts features separately for each context window.
    This enables context-sliced comparison: "Does cooperation have a faster
    build-up rate than rest?"

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned, normalized, and have context_labels set.
        If no context labels, falls back to a single "full" segment.
    window_size : int
        WCC window size in samples.
    hz : float
        Sampling rate.
    onset_threshold : float
        WCC threshold for onset detection.
    max_nan_ratio : float
        Maximum NaN fraction in a segment pair before skipping it entirely.
        Default 0.2 (20%).  Tighter than the old 0.5 threshold to prevent
        structural missingness (e.g., face tracking dropout) from producing
        spurious dynamic feature estimates.

    Returns
    -------
    dict mapping context_label → {"modA_modB": DynamicFeatures, ...}
    If no contexts defined, returns {"full": {pair_key: DynamicFeatures}}.
    """
    feat_cols = dataset.feature_columns
    names = dataset.modality_names
    t_vec = dataset.time_vector()

    # Define segments from context labels
    segments: List[Tuple[str, float, float]] = []
    if dataset.context_labels:
        for ctx in dataset.context_labels:
            segments.append((ctx.label, ctx.start_sec, ctx.end_sec))
    else:
        if len(t_vec) > 0:
            segments.append(("full", t_vec[0], t_vec[-1]))

    if not segments:
        return {}

    # Build segment masks (time-based → sample-index-based)
    results: Dict[str, Dict[str, DynamicFeatures]] = {}

    for label, start_sec, end_sec in segments:
        mask = (t_vec >= start_sec) & (t_vec < end_sec)
        # Minimum segment length: need at least 3 × window_size samples so
        # that the resulting WCC series has >= 2*window_size+1 points — enough
        # for meaningful feature extraction and rolling-origin CV.
        # window_size + 5 is far too lenient (a 45-sample segment at 1 Hz
        # with window_size=10 would pass, but produce only 35 WCC values).
        min_seg_len = 3 * window_size
        if mask.sum() < min_seg_len:
            logger.warning(
                "Context '%s': segment too short (%d samples < %d = 3×window_size). "
                "Skipping — results would lack statistical power.",
                label, int(mask.sum()), min_seg_len,
            )
            results[label] = {}
            continue

        seg_results: Dict[str, DynamicFeatures] = {}
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                for col_a in feat_cols[name_a]:
                    for col_b in feat_cols[name_b]:
                        x = dataset.get_aligned_array(name_a, col_a)
                        y = dataset.get_aligned_array(name_b, col_b)
                        if x is None or y is None:
                            continue

                        # Slice to segment
                        x_seg = x[mask]
                        y_seg = y[mask]

                        # Skip if segment has too many NaNs.
                        # Structural missingness (e.g. face tracking dropout)
                        # causes NaN-filling artefacts in dynamic features.
                        # Require at least (1 - max_nan_ratio) valid samples.
                        valid_ratio = (
                            ~np.isnan(x_seg) & ~np.isnan(y_seg)
                        ).sum() / len(x_seg)
                        if valid_ratio < (1.0 - max_nan_ratio):
                            continue

                        wcc = sliding_window_wcc(
                            x_seg, y_seg, window_size, hz
                        )
                        if len(wcc) < 5:
                            continue
                        feat = extract_dynamic_features(
                            wcc, hz, onset_threshold, max_nan_ratio
                        )
                        key = f"{name_a}_{col_a}__{name_b}_{col_b}"
                        seg_results[key] = feat

        results[label] = seg_results

    return results


# ---------------------------------------------------------------------------
# Multi-peak profile extraction (for long-term baseline analysis)
# ---------------------------------------------------------------------------

def extract_peak_profiles(
    wcc: np.ndarray,
    hz: float = 1.0,
    height: float = 0.2,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Extract multi-peak profiles from a WCC (synchrony) time series.

    Unlike :func:`extract_dynamic_features` (which assumes a single dominant
    sync event per segment), this function detects **all** synchrony peaks
    using `scipy.signal.find_peaks`.  Use this for long-term baseline
    analysis (e.g., 5-min conversation) where the "single-peak fallacy"
    would mischaracterize the data.

    Parameters
    ----------
    wcc : 1-D array
        WCC synchrony time series.
    hz : float
        Sampling rate (for converting samples → seconds).
    height : float
        Minimum WCC value to be considered a peak (default 0.2).
    distance : int or None
        Minimum distance (samples) between peaks.  Default: ``max(10, hz)``.
    prominence : float or None
        Minimum prominence of peaks.  Default: ``height / 2``.

    Returns
    -------
    profiles : list of dict
        Each dict contains per-peak features:
        - ``peak_index``: sample index in ``wcc``
        - ``peak_time_sec``: time in seconds
        - ``peak_value``: WCC value at peak
        - ``prominence``: peak prominence
        - ``width_samples``: peak width (samples, at half prominence)
        - ``inter_peak_interval_sec``: time to next peak (NaN for last)
        - ``build_up_rate``: slope from previous trough to this peak
        - ``breakdown_rate``: slope from this peak to next trough

    Notes
    -----
    This function addresses the **single-peak fallacy** in long recordings.
    For short windows (10-30 s) used in prediction, use
    :func:`extract_dynamic_features` (single-peak, fixed 10 features).

    The output is variable-length (ragged) — caller must aggregate
    (e.g., mean/max/percentile of peak lags) for fixed-dim downstream use.
    """
    if distance is None:
        distance = max(10, int(hz))
    if prominence is None:
        prominence = height / 2.0

    # Handle NaN: replace with local mean for peak detection
    wcc_clean = wcc.copy()
    nan_mask = np.isnan(wcc_clean)
    if nan_mask.any():
        valid_mean = np.nanmean(wcc_clean)
        wcc_clean[nan_mask] = valid_mean if not np.isnan(valid_mean) else 0.0

    # Detect peaks
    peaks, properties = sp_signal.find_peaks(
        wcc_clean,
        height=height,
        distance=distance,
        prominence=prominence,
    )

    if len(peaks) == 0:
        return []

    profiles = []
    n = len(wcc)

    for i, idx in enumerate(peaks):
        # Basic peak info
        prof = {
            "peak_index": int(idx),
            "peak_time_sec": float(idx / hz),
            "peak_value": float(wcc_clean[idx]),
            "prominence": float(properties["prominences"][i]),
            "width_samples": float(properties["widths"][i]) if "widths" in properties else np.nan,
        }

        # Inter-peak interval
        if i < len(peaks) - 1:
            interval = (peaks[i + 1] - idx) / hz
        else:
            interval = np.nan
        prof["inter_peak_interval_sec"] = float(interval)

        # Build-up rate: slope from previous trough to this peak
        if i > 0:
            prev_idx = peaks[i - 1]
            # Find trough between prev_peak and this peak
            segment = wcc_clean[prev_idx:idx + 1]
            trough_idx = prev_idx + np.argmin(segment)
            build_up = (wcc_clean[idx] - wcc_clean[trough_idx]) / ((idx - trough_idx) / hz)
        else:
            # First peak: use first valid WCC as proxy for "pre-sync baseline"
            build_up = (wcc_clean[idx] - np.nanmean(wcc_clean[:max(1, idx)])) / (idx / hz) if idx > 0 else 0.0
        prof["build_up_rate"] = float(build_up)

        # Breakdown rate: slope from this peak to next trough (or end)
        if i < len(peaks) - 1:
            next_idx = peaks[i + 1]
            segment = wcc_clean[idx:next_idx + 1]
            trough_idx = idx + np.argmin(segment)
            breakdown = (wcc_clean[idx] - wcc_clean[trough_idx]) / ((trough_idx - idx) / hz)
        else:
            # Last peak: use post-peak mean as proxy
            post_mean = np.nanmean(wcc_clean[idx:]) if idx < n - 1 else wcc_clean[idx]
            breakdown = (wcc_clean[idx] - post_mean) / ((n - 1 - idx) / hz) if idx < n - 1 else 0.0
        prof["breakdown_rate"] = float(breakdown)

        profiles.append(prof)

    return profiles
