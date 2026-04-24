"""
Within-dyad Z-score Normalization.

Hard-coded preprocessing step. Runs before any dynamic feature extraction.
Each dyad's each indicator is independently Z-transformed so that all
subsequent analyses operate on "standard deviations from baseline" —
the only scale that is comparable across modalities with different
raw units (PLV in [0,1], HRV in [300,1000]ms, Motion Energy in [0,+inf)).
"""

import warnings

import numpy as np
import pandas as pd


def within_dyad_zscore(
    df: pd.DataFrame,
    dyad_col: str = "dyad_id",
    time_col: str | None = None,
    skip_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply within-dyad Z-score normalization to all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format timeseries. Each row = one time point for one dyad.
    dyad_col : str
        Column identifying the dyad.
    time_col : str | None
        Time column (preserved but not normalized).
    skip_cols : list[str] | None
        Additional columns to skip (e.g., context labels).

    Returns
    -------
    normalized : pd.DataFrame
        Z-score normalized copy.
    stats : dict
        Per-dyad, per-column normalization statistics (mean, std)
        for potential re-norm or denormalization.
    """
    if skip_cols is None:
        skip_cols = []
    skip_cols = list(skip_cols) + [dyad_col]
    if time_col is not None:
        skip_cols.append(time_col)

    numeric_cols = [c for c in df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        warnings.warn("No numeric columns found for normalization.")
        return df.copy(), {}

    # Ensure numeric columns are float (avoid LossySetitemError on int columns)
    df = df.copy()
    for col in numeric_cols:
        if not pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float)

    normalized = df
    stats: dict = {}

    for dyad_id in df[dyad_col].unique():
        mask = df[dyad_col] == dyad_id
        stats[dyad_id] = {}
        for col in numeric_cols:
            values = df.loc[mask, col].values.astype(float)

            # Handle NaN
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                normalized.loc[mask, col] = 0.0
                stats[dyad_id][col] = {"mean": 0.0, "std": 1.0, "n_valid": 0}
                continue

            mu = np.mean(valid)
            sigma = np.std(valid, ddof=0)

            if sigma < 1e-12:
                # Constant or near-constant series — skip normalization
                warnings.warn(
                    f"Dyad '{dyad_id}', column '{col}': std ≈ 0 ({sigma:.2e}). "
                    "Skipping normalization for this column."
                )
                normalized.loc[mask, col] = 0.0
                stats[dyad_id][col] = {"mean": float(mu), "std": float(sigma), "n_valid": len(valid), "skipped": True}
                continue

            normalized_values = (values - mu) / sigma
            # Preserve NaN positions
            normalized_values[np.isnan(values)] = np.nan
            normalized.loc[mask, col] = normalized_values
            stats[dyad_id][col] = {
                "mean": float(mu),
                "std": float(sigma),
                "n_valid": len(valid),
                "skipped": False,
            }

    return normalized, stats


def interpolate_short_gaps(
    series: np.ndarray | pd.Series,
    max_gap: int = 3,
) -> np.ndarray:
    """Linear interpolation for short NaN gaps, preserve long gaps.

    Parameters
    ----------
    series : array-like
        1D time series that may contain NaN.
    max_gap : int
        Maximum consecutive NaN epochs to interpolate.

    Returns
    -------
    filled : np.ndarray
        Series with short gaps interpolated, long gaps still NaN.
    """
    arr = np.array(series, dtype=float)
    n = len(arr)
    if n == 0:
        return arr

    result = arr.copy()

    # Find NaN runs
    i = 0
    while i < n:
        if np.isnan(result[i]):
            # Find end of NaN run
            j = i
            while j < n and np.isnan(result[j]):
                j += 1
            gap_len = j - i

            if gap_len <= max_gap:
                # Interpolate if we have valid anchors on both sides
                left = result[i - 1] if i > 0 else np.nan
                right = result[j] if j < n else np.nan
                if not np.isnan(left) and not np.isnan(right):
                    for k in range(gap_len):
                        frac = (k + 1) / (gap_len + 1)
                        result[i + k] = left + frac * (right - left)
                elif not np.isnan(left):
                    for k in range(gap_len):
                        result[i + k] = left
                elif not np.isnan(right):
                    for k in range(gap_len):
                        result[i + k] = right
            # else: leave as NaN (long gap)

            i = j
        else:
            i += 1

    return result
