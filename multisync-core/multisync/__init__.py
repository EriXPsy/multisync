"""
multisync-core — Dynamic process analysis for multimodal interpersonal synchrony.

Architecture: Python Core (computation) + React Web Viewer (visualization).

Quick start (4 lines):

    import multisync as ms
    dyad = ms.Dyad(neural=df_neural, behavioral=df_behavior, hz=1.0)
    dyad.add_context(start=0, end=300, label="Task")
    analyzer = ms.DynamicAnalyzer(window_size=10, surrogate_n=500)
    results = analyzer.fit_transform(dyad)
    results.export_viewer_json("viewer_payload.json")
"""

# High-level API
from .core import Dyad, DynamicAnalyzer, AnalysisResults

# Dataset container
from .dataset import SynchronyDataset, ContextLabel

# Cascade analysis
from .cascade import (
    compute_ccf,
    cascade_analysis,
    rolling_cascade,
    CCAResult,
    CascadeEdge,
)

# Dynamic features
from .dynamic_features import (
    sliding_window_wcc,
    extract_dynamic_features,
    extract_features_all_pairs,
    extract_features_segmented,
    DynamicFeatures,
)

# Prediction
from .prediction import (
    rolling_origin_cv,
    cross_modal_prediction,
    lodo_cv,
    PredictionResult,
    FoldResult,
)

# Synthetic data
from .synthetic import generate_ground_truth_dyad, generate_multimodal_dyad

__all__ = [
    # High-level
    "Dyad",
    "DynamicAnalyzer",
    "AnalysisResults",
    # Dataset
    "SynchronyDataset",
    "ContextLabel",
    # Cascade
    "compute_ccf",
    "cascade_analysis",
    "rolling_cascade",
    "CCAResult",
    "CascadeEdge",
    # Features
    "sliding_window_wcc",
    "extract_dynamic_features",
    "extract_features_all_pairs",
    "extract_features_segmented",
    "DynamicFeatures",
    # Prediction
    "rolling_origin_cv",
    "cross_modal_prediction",
    "lodo_cv",
    "PredictionResult",
    "FoldResult",
    # Synthetic
    "generate_ground_truth_dyad",
    "generate_multimodal_dyad",
]

__version__ = "0.1.0"
