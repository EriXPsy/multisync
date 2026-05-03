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
    extract_features_segmented,
    sliding_window_wcc,
)
from .prediction import FoldResult, PredictionResult, rolling_origin_cv, cross_modal_prediction


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
    # Dynamic features (global, per pair)
    dynamic_features: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Dynamic features (segmented by context)
    dynamic_features_segmented: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # Cascade
    cca_results: List[Dict[str, Any]] = field(default_factory=list)
    cascade_edges: List[Dict[str, Any]] = field(default_factory=list)
    cascade_graph: Dict[str, Any] = field(default_factory=dict)
    # Prediction (nested by modality pair key, e.g. "neural__behavioral")
    prediction: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Cross-modal prediction (source_pair → target_pair)
    cross_modal_prediction: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Cascade graph metrics (in_degree, out_degree, driver_score, etc.)
    cascade_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Context / Score view
    score_view: List[Dict[str, Any]] = field(default_factory=list)
    # Diagnostics — structured log of skipped/failed computations.
    # Each entry: {"stage": str, "pair": str, "reason": str, "detail": dict}
    # This replaces silent logger-only drops so the frontend can render
    # a "Data Exclusion Report" panel instead of showing empty results.
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "0.2.0",
            "dyad_id": self.dyad_id,
            "dynamic_features": self.dynamic_features,
            "dynamic_features_segmented": self.dynamic_features_segmented,
            "cascade_graph": self.cascade_graph,
            "cascade_metrics": self.cascade_metrics,
            "prediction": self.prediction,
            "cross_modal_prediction": self.cross_modal_prediction,
            "score_view": self.score_view,
            "diagnostics": self.diagnostics,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisResults":
        """Deserialize from a dict (inverse of to_dict).
        
        Restores nested objects (DynamicFeatures, PredictionResult, etc.)
        from their dict representations.
        """
        # Restore dynamic_features: Dict[str, Dict] -> Dict[str, DynamicFeatures]
        dyn_feat = {}
        for k, v in d.get("dynamic_features", {}).items():
            if isinstance(v, DynamicFeatures):
                dyn_feat[k] = v
            else:
                dyn_feat[k] = DynamicFeatures.from_dict(v)

        # Restore dynamic_features_segmented: {label: {pair: feat_dict}}
        dyn_seg = {}
        for label, pairs in d.get("dynamic_features_segmented", {}).items():
            dyn_seg[label] = {}
            for pair, feat in pairs.items():
                if isinstance(feat, DynamicFeatures):
                    dyn_seg[label][pair] = feat
                else:
                    dyn_seg[label][pair] = DynamicFeatures.from_dict(feat)

        # Restore prediction: Dict[str, Dict] -> Dict[str, PredictionResult]
        pred = {}
        for k, v in d.get("prediction", {}).items():
            if isinstance(v, PredictionResult):
                pred[k] = v
            else:
                pred[k] = PredictionResult.from_dict(v)

        # Restore cross_modal_prediction
        cross_pred = {}
        for k, v in d.get("cross_modal_prediction", {}).items():
            if isinstance(v, PredictionResult):
                cross_pred[k] = v
            else:
                cross_pred[k] = PredictionResult.from_dict(v)

        # Restore cascade_metrics
        cas_metrics = d.get("cascade_metrics", {})

        # Restore cca_results: List[Dict] -> List[CCAResult]
        cca = []
        for r in d.get("cca_results", []):
            if isinstance(r, CCAResult):
                cca.append(r)
            else:
                cca.append(CCAResult.from_dict(r))

        # Restore cascade_edges: List[Dict] -> List[CascadeEdge]
        edges = []
        for e in d.get("cascade_edges", []):
            if isinstance(e, CascadeEdge):
                edges.append(e)
            else:
                edges.append(CascadeEdge.from_dict(e))

        return cls(
            dyad_id=d.get("dyad_id", "unknown"),
            dynamic_features=dyn_feat,
            dynamic_features_segmented=dyn_seg,
            cca_results=cca,
            cascade_edges=edges,
            cascade_graph=d.get("cascade_graph", {}),
            cascade_metrics=cas_metrics,
            prediction=pred,
            cross_modal_prediction=cross_pred,
            score_view=d.get("score_view", []),
            diagnostics=d.get("diagnostics", []),
            parameters=d.get("parameters", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AnalysisResults":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))

    def export_viewer_json(self, filepath: str) -> str:
        """
        Export viewer-ready JSON.

        This JSON is the single decoupling bridge between Python Core
        and React Viewer.  The Viewer must do ZERO computation — all
        statistics, p-values, peaks, and graph edges are pre-computed.

        Schema:
        {
            "schema_version": "0.2.0",
            "dyad_id": "pair_01",
            "cascade_graph": {
                "nodes": ["Behavior", "Neural"],
                "edges": [{"from": "Behavior", "to": "Neural",
                           "lag_sec": 12.5, "ccf_value": 0.67,
                           "p_value": 0.003, "polarity": "positive"}]
            },
            "dynamic_features": {"behavior__neural": {...}},
            "prediction": {"neural_behavioral": {...}},
            "score_view": [{"start_sec": 0, "end_sec": 300,
                           "label": "Task", "mean_sync": 0.45}],
            "diagnostics": [{"stage": "cascade", "pair": "neural__behavioral",
                             "reason": "segment_too_short", "detail": {...}}]
        }
        """
        d = self.to_dict()

        # Replace NaN/Inf with None (JSON null) before serialization.
        # Also convert numpy scalars and arrays to native Python types so
        # json.dump never raises TypeError on np.float64 / np.ndarray.
        def _sanitize(obj: Any) -> Any:
            # Numpy scalars → native Python
            if isinstance(obj, (np.floating, np.complexfloating)):
                v = float(obj)
                return None if (np.isnan(v) or np.isinf(v)) else v
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return _sanitize(obj.tolist())
            if isinstance(obj, float):
                return None if (np.isnan(obj) or np.isinf(obj)) else obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        d = _sanitize(d)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
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
        1. WCC + 10 dynamic features for each modality pair (global).
        2. WCC + 10 dynamic features segmented by context (if contexts exist).
        3. Cascade analysis (CCF + PRTF surrogate testing).
        4. Prediction window analysis (Rolling Origin CV with dynamic features).
        5. Cross-modal prediction (if 3+ modalities: source pair → target pair).
        6. Score view (context-based synchrony summaries).
        7. Package everything into AnalysisResults.

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
                "prediction_window": self.prediction_window,
                "prediction_horizon": self.prediction_horizon,
                "prediction_gap": self.prediction_gap,
                "hz": hz,
            },
        )

        # 1. Dynamic features (global)
        feat_dict = extract_features_all_pairs(
            dataset,
            window_size=self.window_size,
            hz=hz,
            onset_threshold=self.onset_threshold,
        )
        results.dynamic_features = {k: v.to_dict() for k, v in feat_dict.items()}

        # 2. Dynamic features (context-segmented)
        if dataset.context_labels:
            seg_dict = extract_features_segmented(
                dataset,
                window_size=self.window_size,
                hz=hz,
                onset_threshold=self.onset_threshold,
            )
            results.dynamic_features_segmented = {
                label: {pair: feat.to_dict() for pair, feat in pairs.items()}
                for label, pairs in seg_dict.items()
            }

        # 3. Cascade analysis (CCF + PRTF surrogates)
        cca_results, cascade_edges, cascade_metrics = cascade_analysis(
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
                "polarity": e.polarity,
            }
            for e in cascade_edges
        ]
        results.cascade_graph = {"nodes": nodes, "edges": edges_data}
        results.cascade_metrics = cascade_metrics

        # 4. Prediction window analysis (dynamic features, not raw WCC)
        # Use a larger window for feature extraction (need enough data
        # within each window to compute meaningful dynamic features).
        pred_window = max(self.prediction_window, 30)

        names = dataset.modality_names
        feat_cols = dataset.feature_columns

        # Cache WCC sequences for score view (#6) and reuse
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
                            window_size=pred_window,
                            hz=hz,
                            n_splits=5,
                            gap=max(self.prediction_gap, pred_window // 2),
                            pair_name=pred_key,
                            mode="intra",
                        )
                        if pred.folds:
                            results.prediction[pred_key] = {
                                "modality_a": name_a,
                                "modality_b": name_b,
                                "mode": "intra",
                                "mean_dynamic_auc": pred.mean_dynamic_auc,
                                "mean_baseline_auc": pred.mean_baseline_auc,
                                "mean_delta_auc": pred.mean_delta_auc,
                                "feature_importance": pred.feature_importance,
                                "warning": pred.warning,
                                "n_features_used": pred.n_features_used,
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

        # 5. Cross-modal prediction (if 3+ modalities)
        if len(names) >= 3:
            # Try all ordered source→target combinations
            all_pairs = []
            for i, name_a in enumerate(names):
                for name_b in names[i + 1:]:
                    for col_a in feat_cols[name_a]:
                        for col_b in feat_cols[name_b]:
                            src_key = f"{name_a}_{col_a}__{name_b}_{col_b}"
                            if src_key in wcc_cache:
                                all_pairs.append((src_key, name_a, name_b))

            for idx_a, (src_key, src_a, src_b) in enumerate(all_pairs):
                for idx_b, (tgt_key, tgt_a, tgt_b) in enumerate(all_pairs):
                    if idx_a == idx_b:
                        continue  # skip same pair
                    if src_key in wcc_cache and tgt_key in wcc_cache:
                        cm_pred = cross_modal_prediction(
                            source_wcc=wcc_cache[src_key],
                            target_wcc=wcc_cache[tgt_key],
                            window_size=pred_window,
                            hz=hz,
                            n_splits=3,  # fewer folds for speed
                            gap=max(self.prediction_gap, pred_window // 2),
                            source_name=src_key,
                            target_name=tgt_key,
                        )
                        if cm_pred.folds:
                            cm_key = f"{src_key} -> {tgt_key}"
                            results.cross_modal_prediction[cm_key] = {
                                "source_pair": src_key,
                                "target_pair": tgt_key,
                                "mode": "cross_modal",
                                "mean_dynamic_auc": cm_pred.mean_dynamic_auc,
                                "mean_baseline_auc": cm_pred.mean_baseline_auc,
                                "mean_delta_auc": cm_pred.mean_delta_auc,
                                "feature_importance": cm_pred.feature_importance,
                                "warning": cm_pred.warning,
                                "n_features_used": cm_pred.n_features_used,
                                "folds": [
                                    {
                                        "fold_id": f.fold_id,
                                        "dynamic_auc": f.dynamic_auc,
                                        "baseline_auc": f.baseline_auc,
                                        "delta_auc": f.delta_auc,
                                    }
                                    for f in cm_pred.folds
                                ],
                            }

        # 6. Score view (context-based synchrony summaries)
        if dataset.context_labels:
            t_vec = dataset.time_vector()
            wcc_offset = (self.window_size - 1) / (2.0 * hz)
            for ctx in dataset.context_labels:
                mask = (t_vec >= ctx.start_sec) & (t_vec < ctx.end_sec)
                if not mask.any():
                    continue
                # Map the time-based mask to WCC indices.
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
