"""
High-level API — Dyad and DynamicAnalyzer.

Target: 4 lines of code to go from raw data to viewer-ready JSON.

    import multisync as ms

    # 1. Load and align data
    dyad = ms.Dyad(neural=df_neural, bio_hrv=df_hrv, behavioral=df_motion, hz=1.0)
    # 2. Add context labels
    dyad.add_context(start=0, end=300, label="Task")
    # 3. Analyze
    analyzer = ms.DynamicAnalyzer(window_size=10, surrogate_n=500)
    results = analyzer.fit_transform(dyad)
    # 4. Export
    results.export_viewer_json("viewer_payload.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .cascade import CCAResult, CascadeEdge, cascade_analysis
from .dataset import SynchronyDataset
from .dynamic_features import (
    DynamicFeatures,
    extract_dynamic_features,
    extract_features_all_pairs,
    sliding_window_wcc,
)
from .prediction import FoldResult, PredictionResult, rolling_origin_cv


# ---------------------------------------------------------------------------
# Dyad — thin convenience wrapper around SynchronyDataset
# ---------------------------------------------------------------------------

class Dyad(SynchronyDataset):
    """
    User-friendly dyad container.

    Accepts modality DataFrames as keyword arguments.  Each keyword becomes
    the modality name.

    Parameters
    ----------
    hz : float
        Default target sampling rate for alignment.
    **modalities : DataFrame
        Modality name → DataFrame mapping.
    """

    def __init__(self, hz: float = 1.0, **modalities: pd.DataFrame) -> None:
        # Extract dyad_id if provided as a string; otherwise use default.
        # Always remove it from modalities to prevent add_modality() from
        # treating it as a DataFrame.
        dyad_id = modalities.pop("dyad_id", "dyad_01")
        if not isinstance(dyad_id, str):
            dyad_id = "dyad_01"
        super().__init__(dyad_id=dyad_id)
        self._default_hz = hz
        for name, df in modalities.items():
            self.add_modality(name, df)

    def align(self, target_hz: Optional[float] = None, **kwargs) -> "Dyad":
        hz = target_hz or self._default_hz
        super().align(target_hz=hz, **kwargs)
        return self

    def zscore(self):
        return super().zscore()


# ---------------------------------------------------------------------------
# Analysis results container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResults:
    """Complete analysis output — ready for Viewer JSON export."""
    dyad_id: str
    # Dynamic features
    dynamic_features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Cascade
    cca_results: List[Dict[str, Any]] = field(default_factory=list)
    cascade_edges: List[Dict[str, Any]] = field(default_factory=list)
    cascade_graph: Dict[str, Any] = field(default_factory=dict)
    # Prediction (nested by modality pair key, e.g. "neural__behavioral")
    prediction: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Context / Score view
    score_view: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dyad_id": self.dyad_id,
            "dynamic_features": self.dynamic_features,
            "cascade_graph": self.cascade_graph,
            "prediction": self.prediction,
            "score_view": self.score_view,
            "parameters": self.parameters,
        }

    def export_viewer_json(self, filepath: str) -> str:
        """
        Export viewer-ready JSON.

        This JSON is the single decoupling bridge between Python Core
        and React Viewer.  The Viewer must do ZERO computation — all
        statistics, p-values, peaks, and graph edges are pre-computed.

        Schema:
        {
            "dyad_id": "pair_01",
            "cascade_graph": {
                "nodes": ["Behavior", "Neural"],
                "edges": [{"from": "Behavior", "to": "Neural",
                           "lag_sec": 12.5, "ccf_value": 0.67,
                           "p_value": 0.003}]
            },
            "dynamic_features": {"behavior__neural": {...}},
            "prediction": {"mean_delta_auc": 0.15, "folds": [...]},
            "score_view": [{"start_sec": 0, "end_sec": 300,
                           "label": "Task", "mean_sync": 0.45}]
        }
        """
        d = self.to_dict()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, default=str)
        return str(path)


# ---------------------------------------------------------------------------
# DynamicAnalyzer — the main workhorse
# ---------------------------------------------------------------------------

class DynamicAnalyzer:
    """
    Orchestrates the full analysis pipeline.

    Parameters
    ----------
    window_size : int
        Sliding window size in samples (for WCC and dynamic features).
    surrogate_n : int
        Number of PRTF surrogates for cascade significance testing.
    max_lag_sec : float
        Maximum cross-correlation lag in seconds.
    alpha : float
        Significance threshold for surrogate testing.
    seed : int
        Random seed for reproducibility.
    onset_threshold : float
        WCC threshold for onset detection.
    prediction_window : int
        Window size for prediction features (in samples).
    prediction_horizon : int
        Horizon for prediction labels (in samples).
    prediction_gap : int
        Gap between train and test in prediction CV (in samples).
    """

    def __init__(
        self,
        window_size: int = 10,
        surrogate_n: int = 500,
        max_lag_sec: float = 30.0,
        alpha: float = 0.05,
        seed: int = 42,
        onset_threshold: float = 0.2,
        prediction_window: int = 10,
        prediction_horizon: int = 5,
        prediction_gap: int = 5,
    ) -> None:
        self.window_size = window_size
        self.surrogate_n = surrogate_n
        self.max_lag_sec = max_lag_sec
        self.alpha = alpha
        self.seed = seed
        self.onset_threshold = onset_threshold
        self.prediction_window = prediction_window
        self.prediction_horizon = prediction_horizon
        self.prediction_gap = prediction_gap

    def fit_transform(self, dataset: SynchronyDataset) -> AnalysisResults:
        """
        Run the complete analysis pipeline on an aligned+normalized dataset.

        Steps:
        1. WCC + 10 dynamic features for each modality pair.
        2. Cascade analysis (CCF + PRTF surrogate testing).
        3. Prediction window analysis (Rolling Origin CV).
        4. Score view (context-based synchrony summaries).
        5. Package everything into AnalysisResults.

        Parameters
        ----------
        dataset : SynchronyDataset
            Must already be aligned and normalized.

        Returns
        -------
        AnalysisResults — viewer-ready output.
        """
        if not dataset._aligned:
            raise ValueError("Dataset must be aligned. Call dataset.align() first.")
        if not dataset._normalized:
            raise ValueError(
                "Dataset must be Z-score normalized. Call dataset.zscore() first."
            )

        hz = dataset.target_hz
        results = AnalysisResults(
            dyad_id=dataset.dyad_id,
            parameters={
                "window_size": self.window_size,
                "surrogate_n": self.surrogate_n,
                "max_lag_sec": self.max_lag_sec,
                "alpha": self.alpha,
                "seed": self.seed,
                "onset_threshold": self.onset_threshold,
                "hz": hz,
            },
        )

        # 1. Dynamic features
        feat_dict = extract_features_all_pairs(
            dataset,
            window_size=self.window_size,
            hz=hz,
            onset_threshold=self.onset_threshold,
        )
        results.dynamic_features = {k: v.to_dict() for k, v in feat_dict.items()}

        # 2. Cascade analysis (CCF + PRTF surrogates)
        cca_results, cascade_edges = cascade_analysis(
            dataset,
            max_lag_sec=self.max_lag_sec,
            surrogate_n=self.surrogate_n,
            alpha=self.alpha,
            seed=self.seed,
        )

        # Serialize CCA results
        results.cca_results = []
        for r in cca_results:
            d = {
                "modality_a": r.modality_a,
                "modality_b": r.modality_b,
                "feature_a": r.feature_a,
                "feature_b": r.feature_b,
                "peak_lag_sec": r.peak_lag_sec,
                "peak_ccf": r.peak_ccf,
                "direction": r.direction,
                "is_significant": r.is_significant,
                "p_value": r.p_value,
                "surrogate_n": r.surrogate_n,
            }
            results.cca_results.append(d)

        # Build cascade graph (Viewer-ready)
        nodes = list(set(
            [e.source for e in cascade_edges] + [e.target for e in cascade_edges]
        ))
        edges_data = [
            {
                "from": e.source,
                "to": e.target,
                "lag_sec": e.lag_sec,
                "ccf_value": e.ccf_value,
                "p_value": e.p_value,
                "is_significant": e.is_significant,
            }
            for e in cascade_edges
        ]
        results.cascade_graph = {"nodes": nodes, "edges": edges_data}

        # 3. Prediction window analysis
        # CRITICAL: WCC[t] uses raw data x[t:t+window_size], so it "sees"
        # window_size-1 samples into the future.  The prediction gap must
        # be >= window_size to prevent feature-label data leakage.
        effective_gap = max(self.prediction_gap, self.window_size)
        names = dataset.modality_names
        feat_cols = dataset.feature_columns

        # Cache WCC sequences for score view (#4 fix) and reuse
        wcc_cache: Dict[str, np.ndarray] = {}
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                for col_a in feat_cols[name_a]:
                    for col_b in feat_cols[name_b]:
                        x = dataset.get_aligned_array(name_a, col_a)
                        y = dataset.get_aligned_array(name_b, col_b)
                        if x is None or y is None:
                            continue
                        wcc = sliding_window_wcc(
                            x, y, self.window_size, hz
                        )
                        pred_key = f"{name_a}_{col_a}__{name_b}_{col_b}"
                        wcc_cache[pred_key] = wcc

                        pred = rolling_origin_cv(
                            wcc,
                            window_size=self.prediction_window,
                            horizon=self.prediction_horizon,
                            n_splits=5,
                            gap=effective_gap,
                        )
                        if pred.folds:
                            # Fix #3: Use nested dict to avoid overwriting
                            # across modality pairs
                            results.prediction[pred_key] = {
                                "modality_a": name_a,
                                "modality_b": name_b,
                                "mean_dynamic_auc": pred.mean_dynamic_auc,
                                "mean_baseline_auc": pred.mean_baseline_auc,
                                "mean_delta_auc": pred.mean_delta_auc,
                                "feature_importance": pred.feature_importance,
                                "warning": pred.warning,
                                "folds": [
                                    {
                                        "fold_id": f.fold_id,
                                        "dynamic_auc": f.dynamic_auc,
                                        "baseline_auc": f.baseline_auc,
                                        "delta_auc": f.delta_auc,
                                    }
                                    for f in pred.folds
                                ],
                            }

        # 4. Score view (context-based synchrony summaries)
        # Fix #4: Compute LOCAL mean WCC per context slice, not global mean.
        if dataset.context_labels:
            t_vec = dataset.time_vector()
            wcc_offset = (self.window_size - 1) / (2.0 * hz)
            for ctx in dataset.context_labels:
                mask = (t_vec >= ctx.start_sec) & (t_vec < ctx.end_sec)
                if not mask.any():
                    continue
                # Map the time-based mask to WCC indices.
                # WCC index i corresponds to the window starting at t_vec[i].
                # The window spans [t_vec[i], t_vec[i] + window_size/hz).
                wcc_indices = np.where(mask)[0]
                local_sync_vals = []
                for key, wcc in wcc_cache.items():
                    # Only use WCC indices that fall within valid WCC range
                    valid_idx = wcc_indices[
                        (wcc_indices >= 0) & (wcc_indices < len(wcc))
                    ]
                    if len(valid_idx) == 0:
                        continue
                    local_wcc = wcc[valid_idx]
                    local_mean = np.nanmean(local_wcc)
                    if not np.isnan(local_mean):
                        local_sync_vals.append(local_mean)
                mean_sync = float(np.mean(local_sync_vals)) if local_sync_vals else 0.0
                results.score_view.append({
                    "start_sec": ctx.start_sec,
                    "end_sec": ctx.end_sec,
                    "label": ctx.label,
                    "score": ctx.score,
                    "mean_sync": mean_sync,
                })

        return results
