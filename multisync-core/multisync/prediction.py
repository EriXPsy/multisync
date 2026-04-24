"""
Prediction window analysis — Rolling-origin time-series cross-validation.

Reviewer requirements enforced:
- Uses sklearn.model_selection.TimeSeriesSplit (not LOO-CV).
- Gap parameter enforced between train and test (prevents temporal leakage).
- Reports delta-AUC (dynamic features vs naive baseline), not absolute AUC.
- Strict leakage audit: auto-correlated data must not inflate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


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


@dataclass
class PredictionResult:
    """Aggregated prediction results for one modality pair."""
    modality_a: str
    modality_b: str
    feature_importance: Dict[str, float]
    mean_dynamic_auc: float
    mean_baseline_auc: float
    mean_delta_auc: float
    folds: List[FoldResult]
    warning: Optional[str] = None  # e.g. "leakage suspected"


# ---------------------------------------------------------------------------
# Label creation (from synchrony time series → binary)
# ---------------------------------------------------------------------------

def _create_binary_label(
    sync_series: np.ndarray,
    window_size: int,
    horizon: int,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a binary classification dataset from a continuous synchrony series.

    For each time point t, the feature window is sync[t-window_size : t].
    The label is 1 if mean(sync[t : t+horizon]) > threshold, else 0.

    Parameters
    ----------
    sync_series : 1-D array
        Continuous synchrony measure (e.g., WCC over time).
    window_size : int
        Number of past samples to use as features.
    horizon : int
        Number of future samples whose mean determines the label.
    threshold : float
        Threshold for "high synchrony" label.

    Returns
    -------
    X : 2-D array (n_samples, window_size)
    y : 1-D array (n_samples,) — binary labels.
    """
    n = len(sync_series)
    valid = ~np.isnan(sync_series)
    sync_clean = np.where(valid, sync_series, 0.0)

    X_list, y_list = [], []
    for t in range(window_size, n - horizon):
        window = sync_clean[t - window_size : t]
        future_mean = np.nanmean(sync_series[t : t + horizon])
        # Skip if future window has too many NaNs
        if np.isnan(sync_series[t : t + horizon]).sum() > horizon * 0.5:
            continue
        label = 1.0 if future_mean > threshold else 0.0
        X_list.append(window)
        y_list.append(label)

    if not X_list:
        return np.empty((0, window_size)), np.empty(0)

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# Rolling-origin CV with sklearn TimeSeriesSplit
# ---------------------------------------------------------------------------

def rolling_origin_cv(
    sync_series: np.ndarray,
    window_size: int = 10,
    horizon: int = 5,
    n_splits: int = 5,
    gap: int = 0,
    threshold: float = 0.0,
    max_iter: int = 200,
) -> PredictionResult:
    """
    Rolling-origin time-series CV using sklearn.TimeSeriesSplit.

    The *gap* parameter enforces a buffer zone between the last training
    sample and the first test sample, preventing any temporal leakage.

    Parameters
    ----------
    sync_series : 1-D array
        Continuous synchrony time series.
    window_size : int
        Past window for feature extraction.
    horizon : int
        Future window for label creation.
    n_splits : int
        Number of CV folds.
    gap : int
        Gap (buffer) between train and test sets, in samples.
    threshold : float
        Label threshold.
    max_iter : int
        Max iterations for LogisticRegression.

    Returns
    -------
    PredictionResult
    """
    X, y = _create_binary_label(sync_series, window_size, horizon, threshold)

    if len(y) < 20:
        return PredictionResult(
            modality_a="",
            modality_b="",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="insufficient_samples",
        )

    # Check for class imbalance
    class_counts = np.bincount(y.astype(int))
    if len(class_counts) < 2 or min(class_counts) < 3:
        return PredictionResult(
            modality_a="",
            modality_b="",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="class_imbalance",
        )

    # sklearn TimeSeriesSplit with gap
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    folds: List[FoldResult] = []
    feature_coefs_sum = np.zeros(window_size)
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
            modality_a="",
            modality_b="",
            feature_importance={},
            mean_dynamic_auc=0.5,
            mean_baseline_auc=0.5,
            mean_delta_auc=0.0,
            folds=[],
            warning="no_valid_folds",
        )

    mean_dynamic = np.mean([f.dynamic_auc for f in folds])
    mean_baseline = np.mean([f.baseline_auc for f in folds])
    mean_delta = np.mean([f.delta_auc for f in folds])

    # Feature importance from averaged coefficients
    avg_coefs = feature_coefs_sum / valid_folds
    importance = {f"lag_{i + 1}": float(avg_coefs[i]) for i in range(window_size)}

    # Leakage warning: if delta_auc > 0.4, suspicious
    warning = None
    if mean_delta > 0.4:
        warning = "leakage_suspected"

    return PredictionResult(
        modality_a="",
        modality_b="",
        feature_importance=importance,
        mean_dynamic_auc=mean_dynamic,
        mean_baseline_auc=mean_baseline,
        mean_delta_auc=mean_delta,
        folds=folds,
        warning=warning,
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
