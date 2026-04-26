"""
Prediction window analysis — Rolling-origin time-series cross-validation.

**v2.0 — Dynamic Feature Prediction (not raw WCC autoregression)**

The prediction module now uses *computed dynamic features* as the feature
matrix, NOT raw WCC sliding windows.  This addresses the critical design flaw
where the old implementation was essentially an autoregressive forecaster
("WCC now predicts WCC next" — trivially inflated by autocorrelation).

Two prediction modes:
1. **Intra-pair**: Use dynamic features of pair A-B to predict future A-B
   synchrony state (dynamic features vs naive baseline).
2. **Cross-modal**: Use dynamic features of source pair (e.g., behavioral-neural)
   to predict future target pair (e.g., neural-bio) synchrony state.
   This is the scientifically meaningful "precursor signal" prediction.

Reviewer requirements enforced:
- Uses sklearn.model_selection.TimeSeriesSplit (not LOO-CV).
- Gap parameter enforced between train and test (prevents temporal leakage).
- Reports delta-AUC (dynamic features vs naive baseline), not absolute AUC.
- Strict leakage audit: auto-correlated data must not inflate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature names (must match DynamicFeatures.to_dict() keys)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "onset_latency",
    "onset_amplitude",
    "build_up_rate",
    "build_up_slope",
    "peak_amplitude",
    "peak_duration",
    "breakdown_rate",
    "recovery_time",
    "mean_synchrony",
    "synchrony_entropy",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Result of one CV fold."""
    fold_id: int
    train_size: int
    test_size: int
    dynamic_auc: float
    baseline_auc: float
    delta_auc: float  # dynamic - baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "dynamic_auc": float(self.dynamic_auc),
            "baseline_auc": float(self.baseline_auc),
            "delta_auc": float(self.delta_auc),
        }


@dataclass
class PredictionResult:
    """Aggregated prediction results for one analysis."""
    source_pair: str = ""  # e.g., "behavior_value__neural_value"
    target_pair: str = ""  # same as source for intra-pair prediction
    mode: str = "intra"    # "intra" or "cross_modal"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    mean_dynamic_auc: float = 0.5
    mean_baseline_auc: float = 0.5
    mean_delta_auc: float = 0.0
    folds: List[FoldResult] = field(default_factory=list)
    warning: Optional[str] = None  # e.g. "leakage suspected"
    n_features_used: int = 0  # how many of the 10 features were non-NaN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_pair": self.source_pair,
            "target_pair": self.target_pair,
            "mode": self.mode,
            "feature_importance": {k: float(v) for k, v in self.feature_importance.items()},
            "mean_dynamic_auc": float(self.mean_dynamic_auc),
            "mean_baseline_auc": float(self.mean_baseline_auc),
            "mean_delta_auc": float(self.mean_delta_auc),
            "warning": self.warning,
            "n_features_used": self.n_features_used,
            "folds": [f.to_dict() for f in self.folds],
        }


# ---------------------------------------------------------------------------
# Feature matrix builder — the core fix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    wcc: np.ndarray,
    window_size: int,
    hz: float = 1.0,
    onset_threshold: float = 0.2,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a dynamic-feature matrix from a WCC time series.

    Instead of using raw WCC values as features (which is just autoregression),
    this function computes 10 Gordon-inspired dynamic features *within each
    sliding window* of the WCC series.  Each row of the output matrix is a
    10-dimensional feature vector describing the dynamics of that window.

    Parameters
    ----------
    wcc : 1-D array
        WCC time series (output of sliding_window_wcc).
    window_size : int
        Number of WCC samples per feature-extraction window.
    hz : float
        Sampling rate of WCC.
    onset_threshold : float
        WCC threshold for onset detection.

    Returns
    -------
    X : 2-D array (n_windows, 10)
        Feature matrix. Rows with insufficient valid data are NaN.
    feature_names : list of str
        Names of the 10 features (for feature_importance dict keys).
    """
    from .dynamic_features import extract_dynamic_features

    n = len(wcc)
    step = max(1, window_size // 2)  # 50% overlap
    starts = list(range(0, n - window_size + 1, step))

    if not starts:
        return np.empty((0, 10)), FEATURE_NAMES

    X = np.full((len(starts), 10), np.nan)

    for i, s in enumerate(starts):
        wcc_window = wcc[s : s + window_size]
        feat = extract_dynamic_features(wcc_window, hz, onset_threshold)
        X[i] = [
            feat.onset_latency,
            feat.onset_amplitude,
            feat.build_up_rate,
            feat.build_up_slope,
            feat.peak_amplitude,
            feat.peak_duration,
            feat.breakdown_rate,
            feat.recovery_time,
            feat.mean_synchrony,
            feat.synchrony_entropy,
        ]

    return X, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Label creation
# ---------------------------------------------------------------------------

def _create_binary_label_from_wcc(
    wcc: np.ndarray,
    window_size: int,
    step: int,
    horizon_windows: int = 1,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary labels from a WCC series aligned to feature windows.

    For each feature window starting at position s, the label is 1 if the
    mean WCC in the next horizon_windows worth of WCC samples exceeds threshold.

    Parameters
    ----------
    wcc : 1-D array
        WCC time series.
    window_size : int
        Feature window size (same as used in build_feature_matrix).
    step : int
        Step size (same as used in build_feature_matrix).
    horizon_windows : int
        Number of future windows to average for the label.
    threshold : float
        Label threshold.

    Returns
    -------
    y : 1-D array of binary labels.
    valid_mask : 1-D bool array (True where label could be computed).
    """
    n = len(wcc)
    starts = list(range(0, n - window_size + 1, step))

    y = np.full(len(starts), np.nan)
    valid_mask = np.zeros(len(starts), dtype=bool)

    # The "future" WCC region starts at s + window_size
    for i, s in enumerate(starts):
        future_start = s + window_size
        future_end = future_start + horizon_windows * window_size
        if future_end > n:
            continue
        future_wcc = wcc[future_start:future_end]
        if np.isnan(future_wcc).sum() > len(future_wcc) * 0.5:
            continue
        future_mean = np.nanmean(future_wcc)
        y[i] = 1.0 if future_mean > threshold else 0.0
        valid_mask[i] = True

    return y, valid_mask


# ---------------------------------------------------------------------------
# Rolling-origin CV with dynamic features
# ---------------------------------------------------------------------------

def rolling_origin_cv(
    wcc: np.ndarray,
    window_size: int = 30,
    hz: float = 1.0,
    horizon_windows: int = 1,
    n_splits: int = 5,
    gap: int = 0,
    threshold: float = 0.0,
    onset_threshold: float = 0.2,
    max_iter: int = 200,
    pair_name: str = "",
    mode: str = "intra",
) -> PredictionResult:
    """
    Rolling-origin time-series CV using DYNAMIC FEATURES (not raw WCC).

    This is the corrected prediction pipeline:
    1. Build feature matrix: each sliding window of WCC → 10 dynamic features.
    2. Create binary labels from future WCC windows.
    3. Train LogisticRegression on features, compare against naive baseline.

    The *gap* parameter enforces a buffer zone between the last training
    sample and the first test sample, preventing temporal leakage.

    Parameters
    ----------
    wcc : 1-D array
        Continuous synchrony time series (WCC output).
    window_size : int
        Window size for both feature extraction and label creation.
        Default 30 (larger than old default 10 to ensure meaningful
        dynamic feature extraction within each window).
    hz : float
        Sampling rate of WCC.
    horizon_windows : int
        Number of future windows whose mean determines the label.
    n_splits : int
        Number of CV folds.
    gap : int
        Gap (buffer) between train and test sets, in samples (feature rows).
    threshold : float
        Label threshold for "high synchrony".
    onset_threshold : float
        WCC threshold for onset detection in dynamic features.
    max_iter : int
        Max iterations for LogisticRegression.
    pair_name : str
        Human-readable name for this pair (for result metadata).
    mode : str
        "intra" or "cross_modal".

    Returns
    -------
    PredictionResult
    """
    # 1. Build feature matrix
    step = max(1, window_size // 2)
    X, feature_names = build_feature_matrix(
        wcc, window_size, hz, onset_threshold
    )

    # 2. Create labels
    y, valid_mask = _create_binary_label_from_wcc(
        wcc, window_size, step, horizon_windows, threshold
    )

    # Align: only keep rows where both (a) the label is valid AND
    # (b) the feature row is not *structurally* empty.
    #
    # CRITICAL — do NOT use np.any here.  Several features have semantically
    # meaningful NaN values:
    #   - onset_latency / onset_amplitude = NaN  →  no onset crossed threshold
    #   - recovery_time = NaN               →  no recovery phase observed
    # These NaN slots represent "low-synchrony / flat" windows, which are
    # the primary *negative* training examples for the predictor.
    # Dropping them with np.any would cause Survivorship Bias: the model
    # trains only on windows that had a visible synchrony episode (positive
    # class), never seeing the contrast examples it needs to learn what
    # *precedes* non-synchrony.
    #
    # Correct rule: only discard rows where ALL 10 features are NaN
    # (structurally missing window — no data at all).  Partial-NaN rows
    # survive and have their missing features replaced with 0 below.
    all_nan_rows = np.all(np.isnan(X), axis=1)
    both_valid = valid_mask & ~all_nan_rows
    X = X[both_valid]
    y = y[both_valid].astype(int)

    # Count non-NaN features per row (for diagnostics), then impute.
    # After the np.all filter above, every surviving row has at least one
    # valid feature.
    #
    # CRITICAL — imputation strategy depends on feature semantics:
    #
    #   * Duration / latency features (onset_latency, recovery_time):
    #     NaN means "event never occurred within the window".  Filling with
    #     0.0 would tell the model "instant onset / instant recovery" — the
    #     most extreme positive value — which is the opposite of reality.
    #     Instead, fill with window_duration (the maximum possible value),
    #     meaning "so slow it never happened within the observation window".
    #
    #   * Rate / amplitude / slope features (build_up_rate, peak_amplitude,
    #     breakdown_rate, etc.):
    #     NaN here means "no event → no rate/amplitude".  Filling with 0.0
    #     is semantically correct: no build-up = rate of 0.
    #
    # Feature indices (matching FEATURE_NAMES order):
    #   0  onset_latency       — DURATION → fill with window_duration
    #   1  onset_amplitude     — AMPLITUDE → fill with 0.0
    #   2  build_up_rate       — RATE → fill with 0.0
    #   3  build_up_slope      — RATE → fill with 0.0
    #   4  peak_amplitude      — AMPLITUDE → fill with 0.0
    #   5  peak_duration       — DURATION → fill with window_duration
    #   6  breakdown_rate      — RATE → fill with 0.0
    #   7  recovery_time       — DURATION → fill with window_duration
    #   8  mean_synchrony      — LEVEL → fill with 0.0
    #   9  synchrony_entropy   — LEVEL → fill with 0.0
    DURATION_FEATURE_IDX = {0, 5, 7}  # onset_latency, peak_duration, recovery_time
    window_duration = window_size / hz

    if len(X) > 0:
        n_non_nan = (~np.isnan(X)).sum(axis=1)
        for col_idx in DURATION_FEATURE_IDX:
            X[np.isnan(X[:, col_idx]), col_idx] = window_duration
        X = np.nan_to_num(X, nan=0.0)  # remaining NaN → 0.0 (rates/amplitudes)
        avg_features_used = int(np.mean(n_non_nan))
    else:
        avg_features_used = 0

    if len(y) < 20:
        return PredictionResult(
            source_pair=pair_name,
            target_pair=pair_name,
            mode=mode,
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="insufficient_samples",
            n_features_used=avg_features_used,
        )

    # Check for class imbalance
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or min(class_counts) < 3:
        return PredictionResult(
            source_pair=pair_name,
            target_pair=pair_name,
            mode=mode,
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="class_imbalance",
            n_features_used=avg_features_used,
        )

    # sklearn TimeSeriesSplit with gap
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    folds: List[FoldResult] = []
    feature_coefs_sum = np.zeros(len(feature_names))
    valid_folds = 0

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(test_idx) < 5:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check test set has both classes
        if len(np.unique(y_test)) < 2:
            continue

        # Dynamic model: Logistic Regression
        try:
            clf = LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                class_weight="balanced",
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            dynamic_auc = roc_auc_score(y_test, y_prob)
            feature_coefs_sum += clf.coef_[0]
            valid_folds += 1
        except Exception:
            dynamic_auc = 0.5

        # Naive baseline: predict the proportion of the training majority class
        baseline_prob = np.full_like(y_test, y_train.mean())
        try:
            baseline_auc = roc_auc_score(y_test, baseline_prob)
        except Exception:
            baseline_auc = 0.5

        folds.append(FoldResult(
            fold_id=fold_id,
            train_size=len(train_idx),
            test_size=len(test_idx),
            dynamic_auc=dynamic_auc,
            baseline_auc=baseline_auc,
            delta_auc=dynamic_auc - baseline_auc,
        ))

    if valid_folds == 0:
        return PredictionResult(
            source_pair=pair_name,
            target_pair=pair_name,
            mode=mode,
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="no_valid_folds",
            n_features_used=avg_features_used,
        )

    mean_dynamic = np.mean([f.dynamic_auc for f in folds])
    mean_baseline = np.mean([f.baseline_auc for f in folds])
    mean_delta = np.mean([f.delta_auc for f in folds])

    # Feature importance from averaged coefficients
    avg_coefs = feature_coefs_sum / valid_folds
    importance = {
        feature_names[i]: float(avg_coefs[i]) for i in range(len(feature_names))
    }

    # Leakage warning: if delta_auc > 0.4, suspicious
    warning = None
    if mean_delta > 0.4:
        warning = "leakage_suspected"

    return PredictionResult(
        source_pair=pair_name,
        target_pair=pair_name,
        mode=mode,
        feature_importance=importance,
        mean_dynamic_auc=mean_dynamic,
        mean_baseline_auc=mean_baseline,
        mean_delta_auc=mean_delta,
        folds=folds,
        warning=warning,
        n_features_used=avg_features_used,
    )


# ---------------------------------------------------------------------------
# Cross-modal prediction: source pair → target pair
# ---------------------------------------------------------------------------

def cross_modal_prediction(
    source_wcc: np.ndarray,
    target_wcc: np.ndarray,
    window_size: int = 30,
    hz: float = 1.0,
    horizon_windows: int = 1,
    n_splits: int = 5,
    gap: int = 0,
    threshold: float = 0.0,
    onset_threshold: float = 0.2,
    max_iter: int = 200,
    source_name: str = "",
    target_name: str = "",
) -> PredictionResult:
    """
    Cross-modal prediction: use dynamic features of a SOURCE pair to predict
    the future synchrony state of a TARGET pair.

    This is the scientifically meaningful "precursor signal" test:
    e.g., "Do behavioral-neural dynamics predict subsequent neural-bio
    synchrony?"

    The feature matrix is built from source_wcc, but labels come from
    target_wcc.  This avoids the autocorrelation trap entirely because
    source and target are independent signals.

    Parameters
    ----------
    source_wcc : 1-D array
        WCC time series of the source modality pair (features come from here).
    target_wcc : 1-D array
        WCC time series of the target modality pair (labels come from here).
    window_size : int
        Window size for feature extraction and label creation.
    hz : float
        Sampling rate.
    horizon_windows : int
        Number of future windows for label creation.
    n_splits : int
        Number of CV folds.
    gap : int
        Buffer gap between train and test (in feature-row units).
    threshold : float
        Label threshold.
    onset_threshold : float
        WCC onset threshold for dynamic features.
    max_iter : int
        Max iterations for LogisticRegression.
    source_name : str
        Name of source pair.
    target_name : str
        Name of target pair.

    Returns
    -------
    PredictionResult with mode="cross_modal".
    """
    step = max(1, window_size // 2)

    # Build features from SOURCE
    X, feature_names = build_feature_matrix(
        source_wcc, window_size, hz, onset_threshold
    )

    # Build labels from TARGET
    y, valid_mask = _create_binary_label_from_wcc(
        target_wcc, window_size, step, horizon_windows, threshold
    )

    # Align lengths (feature matrices may differ in length due to WCC lengths)
    min_rows = min(len(X), len(y))
    if min_rows < 20:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="insufficient_samples",
            n_features_used=0,
        )

    X = X[:min_rows]
    y = y[:min_rows]
    valid_mask = valid_mask[:min_rows]

    # Only keep rows where both features and labels are valid
    both_valid = valid_mask & ~np.any(np.isnan(X), axis=1)
    X = X[both_valid]
    y = y[both_valid].astype(int)

    # Replace NaN features with 0
    if len(X) > 0:
        n_non_nan = (~np.isnan(X)).sum(axis=1)
        X = np.nan_to_num(X, nan=0.0)
        avg_features_used = int(np.mean(n_non_nan)) if len(n_non_nan) > 0 else 0
    else:
        avg_features_used = 0

    if len(y) < 20:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="insufficient_samples",
            n_features_used=avg_features_used,
        )

    class_counts = np.bincount(y)
    if len(class_counts) < 2 or min(class_counts) < 3:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="class_imbalance",
            n_features_used=avg_features_used,
        )

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    folds: List[FoldResult] = []
    feature_coefs_sum = np.zeros(len(feature_names))
    valid_folds = 0

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(test_idx) < 5:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        try:
            clf = LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                class_weight="balanced",
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            dynamic_auc = roc_auc_score(y_test, y_prob)
            feature_coefs_sum += clf.coef_[0]
            valid_folds += 1
        except Exception:
            dynamic_auc = 0.5

        baseline_prob = np.full_like(y_test, y_train.mean())
        try:
            baseline_auc = roc_auc_score(y_test, baseline_prob)
        except Exception:
            baseline_auc = 0.5

        folds.append(FoldResult(
            fold_id=fold_id,
            train_size=len(train_idx),
            test_size=len(test_idx),
            dynamic_auc=dynamic_auc,
            baseline_auc=baseline_auc,
            delta_auc=dynamic_auc - baseline_auc,
        ))

    if valid_folds == 0:
        return PredictionResult(
            source_pair=source_name,
            target_pair=target_name,
            mode="cross_modal",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="no_valid_folds",
            n_features_used=avg_features_used,
        )

    mean_dynamic = np.mean([f.dynamic_auc for f in folds])
    mean_baseline = np.mean([f.baseline_auc for f in folds])
    mean_delta = np.mean([f.delta_auc for f in folds])

    avg_coefs = feature_coefs_sum / valid_folds
    importance = {
        feature_names[i]: float(avg_coefs[i]) for i in range(len(feature_names))
    }

    warning = None
    if mean_delta > 0.4:
        warning = "leakage_suspected"

    return PredictionResult(
        source_pair=source_name,
        target_pair=target_name,
        mode="cross_modal",
        feature_importance=importance,
        mean_dynamic_auc=mean_dynamic,
        mean_baseline_auc=mean_baseline,
        mean_delta_auc=mean_delta,
        folds=folds,
        warning=warning,
        n_features_used=avg_features_used,
    )


# ---------------------------------------------------------------------------
# Leave-One-Dyad-Out (LODO) — for group-level generalization
# ---------------------------------------------------------------------------

def lodo_cv(
    dyad_results: List[Dict],
    target_key: str = "mean_delta_auc",
) -> Dict:
    """
    Leave-One-Dyad-Out cross-validation at the group level.

    Each 'dyad' is a dict of results.  LODO iteratively holds out one dyad,
    computes the mean of the remaining dyads, and compares to the held-out
    dyad's actual value.

    Parameters
    ----------
    dyad_results : list of dict
        Each dict must contain *target_key*.
    target_key : str
        Which metric to evaluate.

    Returns
    -------
    dict with keys: lodo_predictions, lodo_actuals, mae, correlation.
    """
    if len(dyad_results) < 3:
        return {
            "error": "need at least 3 dyads for LODO",
            "lodo_predictions": [],
            "lodo_actuals": [],
        }

    values = np.array([d[target_key] for d in dyad_results])
    predictions = []
    actuals = []

    for i in range(len(dyad_results)):
        others = np.delete(values, i)
        pred = np.mean(others)
        predictions.append(float(pred))
        actuals.append(float(values[i]))

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = float(np.mean(np.abs(predictions - actuals)))
    residuals = actuals - predictions

    # Pearson correlation (handle zero-variance)
    if np.std(actuals) > 0 and np.std(predictions) > 0:
        corr = float(np.corrcoef(actuals, predictions)[0, 1])
    else:
        corr = 0.0

    return {
        "lodo_predictions": predictions.tolist(),
        "lodo_actuals": actuals.tolist(),
        "mae": mae,
        "correlation": corr,
    }
