"""
SynchronyDataset — Multi-modal dyadic data container.

Handles: multi-Hz alignment, within-dyad Z-score, NaN imputation,
and context (episode/score) annotation.

Design target: replace fragile CSV wrangling with a single typed object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate as sp_interp


# ---------------------------------------------------------------------------
# Context annotation
# ---------------------------------------------------------------------------

@dataclass
class ContextLabel:
    """A scored / labelled episode annotation (the psycho context layer)."""
    start_sec: float
    end_sec: float
    label: str
    score: float = 0.0  # optional continuous score (e.g. rapport rating)


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------

class SynchronyDataset:
    """
    Container for one dyad's multi-modal time-series.

    Each modality is stored as a ``pandas.DataFrame`` with a ``time`` column
    (seconds) and one or more feature columns.  Modalities may have different
    native sampling rates; :meth:`align` resamples everything to a common rate.

    Parameters
    ----------
    dyad_id : str
        Identifier for this dyad pair.
    modalities : dict[str, DataFrame]
        Mapping of modality name → DataFrame.  Each DataFrame must contain a
        ``time`` column (monotonically increasing, seconds) plus at least one
        numeric feature column.
    """

    def __init__(
        self,
        dyad_id: str,
        modalities: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.dyad_id = dyad_id
        self.modalities: Dict[str, pd.DataFrame] = {}
        self.context_labels: List[ContextLabel] = []
        self._aligned: bool = False
        self._normalized: bool = False
        self.target_hz: float = 1.0

        if modalities:
            for name, df in modalities.items():
                self.add_modality(name, df)

    # ------------------------------------------------------------------
    # Modality management
    # ------------------------------------------------------------------

    def add_modality(self, name: str, df: pd.DataFrame) -> "SynchronyDataset":
        """Register a modality DataFrame.  Must contain a 'time' column."""
        df = df.copy()
        if "time" not in df.columns:
            raise ValueError(
                f"Modality '{name}' must have a 'time' column. "
                f"Got columns: {list(df.columns)}"
            )
        if not pd.api.types.is_numeric_dtype(df["time"]):
            raise ValueError(f"'time' column in '{name}' must be numeric.")

        # Sort by time, drop duplicate timestamps
        df = df.sort_values("time").drop_duplicates(subset="time")
        df = df.reset_index(drop=True)

        self.modalities[name] = df
        self._aligned = False
        return self

    @property
    def modality_names(self) -> List[str]:
        return list(self.modalities.keys())

    @property
    def feature_columns(self) -> Dict[str, List[str]]:
        """Return {modality: [feature_cols]} for each modality."""
        out: Dict[str, List[str]] = {}
        for name, df in self.modalities.items():
            cols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
            # Ensure float dtype for downstream ops
            for c in cols:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype(float)
            out[name] = cols
        return out

    # ------------------------------------------------------------------
    # Context annotation (psycho layer)
    # ------------------------------------------------------------------

    def add_context(
        self, start: float, end: float, label: str, score: float = 0.0
    ) -> "SynchronyDataset":
        """Add a scored episode annotation."""
        if start >= end:
            raise ValueError(f"start ({start}) must be < end ({end}).")
        self.context_labels.append(ContextLabel(start, end, label, score))
        # Sort by start time
        self.context_labels.sort(key=lambda c: c.start_sec)
        return self

    def get_context_at(self, t: float) -> Optional[ContextLabel]:
        """Return the context label active at time *t*, if any."""
        for ctx in self.context_labels:
            if ctx.start_sec <= t < ctx.end_sec:
                return ctx
        return None

    # ------------------------------------------------------------------
    # Alignment — resample all modalities to a common rate
    # ------------------------------------------------------------------

    def align(
        self,
        target_hz: float = 1.0,
        method: str = "linear",
    ) -> "SynchronyDataset":
        """
        Resample every modality to *target_hz* using linear (default) or
        nearest-neighbour interpolation.  The common time axis spans the
        intersection of all modalities' ranges.

        Parameters
        ----------
        target_hz : float
            Target sampling rate in Hz.  Default 1.0 (1 sample/second).
        method : str
            'linear' | 'nearest' | 'cubic'.  Passed to scipy.interpolate.
        """
        if not self.modalities:
            raise ValueError("No modalities to align.")

        self.target_hz = target_hz
        feat_cols = self.feature_columns

        # Common time span = intersection of all modality ranges
        t_starts = [df["time"].iloc[0] for df in self.modalities.values()]
        t_ends = [df["time"].iloc[-1] for df in self.modalities.values()]
        t_min = max(t_starts)
        t_max = min(t_ends)

        if t_min >= t_max:
            raise ValueError(
                "Modalities have no overlapping time range. "
                f"Ranges: {dict(zip(self.modality_names, zip(t_starts, t_ends)))}"
            )

        n_samples = int(np.floor((t_max - t_min) * target_hz)) + 1
        common_time = np.linspace(t_min, t_max, n_samples)

        for name, df in self.modalities.items():
            cols = feat_cols[name]
            original_time = df["time"].values.astype(float)

            new_df = pd.DataFrame({"time": common_time})
            for col in cols:
                valid = ~np.isnan(df[col].values)
                if valid.sum() < 2:
                    # Not enough data to interpolate; fill with NaN
                    new_df[col] = np.nan
                    continue

                if method == "nearest":
                    kind = "nearest"
                elif method == "cubic" and valid.sum() >= 4:
                    kind = "cubic"
                else:
                    kind = "linear"

                interp_func = sp_interp.interp1d(
                    original_time[valid],
                    df[col].values[valid],
                    kind=kind,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                new_df[col] = interp_func(common_time)

            self.modalities[name] = new_df

        self._aligned = True
        return self

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------

    def handle_nan(
        self,
        strategy: str = "ffill",
        max_gap_sec: Optional[float] = None,
    ) -> "SynchronyDataset":
        """
        Fill or drop NaN values.

        Parameters
        ----------
        strategy : str
            'ffill' — forward fill (default)
            'drop_window' — drop windows where ANY modality has NaN
            'interpolate' — linear interpolation within gaps
        max_gap_sec : float or None
            If set, NaN gaps longer than this (seconds) are NOT filled
            and remain NaN (prevents imputing over long signal dropouts).
        """
        if not self._aligned:
            warnings.warn(
                "Data not yet aligned. Call align() first for reliable results.",
                UserWarning,
            )

        dt = 1.0 / self.target_hz if self.target_hz > 0 else 1.0
        max_gap_samples = int(max_gap_sec / dt) if max_gap_sec else None

        feat_cols = self.feature_columns

        for name in self.modality_names:
            df = self.modalities[name]
            for col in feat_cols[name]:
                series = df[col].copy()

                if strategy == "ffill":
                    if max_gap_samples:
                        # Only ffill gaps <= max_gap_samples
                        nan_groups = (series.isna() != series.isna().shift()).cumsum()
                        for grp_id in nan_groups[series.isna()].unique():
                            mask = nan_groups == grp_id
                            if mask.sum() <= max_gap_samples:
                                series[mask] = series.ffill()[mask]
                            # else: leave as NaN
                    else:
                        series = series.ffill().bfill()

                elif strategy == "interpolate":
                    if max_gap_samples:
                        nan_groups = (series.isna() != series.isna().shift()).cumsum()
                        for grp_id in nan_groups[series.isna()].unique():
                            mask = nan_groups == grp_id
                            if mask.sum() <= max_gap_samples:
                                series[mask] = series.interpolate(method="linear")[mask]
                    else:
                        series = series.interpolate(method="linear").ffill().bfill()

                elif strategy == "drop_window":
                    # Mark rows where ANY feature in ANY modality is NaN
                    pass  # handled below

                else:
                    raise ValueError(f"Unknown NaN strategy: {strategy}")

                df[col] = series
            self.modalities[name] = df

        if strategy == "drop_window":
            # Build a valid mask across all modalities
            common_len = len(next(iter(self.modalities.values())))
            valid_mask = np.ones(common_len, dtype=bool)
            for name in self.modality_names:
                df = self.modalities[name]
                for col in feat_cols[name]:
                    valid_mask &= ~df[col].isna().values
            # Trim all modalities to valid rows
            for name in self.modality_names:
                self.modalities[name] = self.modalities[name][valid_mask].reset_index(drop=True)

        return self

    # ------------------------------------------------------------------
    # Within-dyad Z-score normalization
    # ------------------------------------------------------------------

    def zscore(self) -> Tuple["SynchronyDataset", Dict]:
        """
        Within-dyad Z-score normalization (ddof=0).  Each feature is
        independently standardized across all time points in this dyad.

        Returns
        -------
        self (in-place) and a dict of pre-normalization statistics.
        """
        feat_cols = self.feature_columns
        stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        for name in self.modality_names:
            df = self.modalities[name]
            stats[name] = {}
            for col in feat_cols[name]:
                mu = float(df[col].mean())
                sigma = float(df[col].std(ddof=0))
                stats[name][col] = {"mean": mu, "std": sigma}
                if sigma > 0:
                    df[col] = (df[col] - mu) / sigma
                else:
                    df[col] = 0.0

            self.modalities[name] = df

        self._normalized = True
        return self, stats

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_aligned_array(
        self, modality: str, feature: str
    ) -> Optional[np.ndarray]:
        """Return a 1-D numpy array for one modality+feature."""
        if modality not in self.modalities:
            return None
        df = self.modalities[modality]
        if feature not in df.columns:
            return None
        return df[feature].values.astype(float)

    def time_vector(self) -> np.ndarray:
        """Return the common time vector (requires prior align())."""
        if not self.modalities:
            return np.array([])
        # All modalities share the same time after align()
        return self.modalities[next(iter(self.modalities))]["time"].values

    def summary(self) -> str:
        lines = [f"SynchronyDataset '{self.dyad_id}'"]
        lines.append(f"  Aligned: {self._aligned} | Normalized: {self._normalized}")
        lines.append(f"  Target Hz: {self.target_hz}")
        for name, df in self.modalities.items():
            cols = self.feature_columns.get(name, [])
            n_nan = sum(df[c].isna().sum() for c in cols)
            lines.append(
                f"  {name}: {len(df)} samples, {len(cols)} features, "
                f"{n_nan} NaNs, t=[{df['time'].iloc[0]:.1f}, {df['time'].iloc[-1]:.1f}]s"
            )
        lines.append(f"  Context labels: {len(self.context_labels)}")
        return "\n".join(lines)
