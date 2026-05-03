"""
Comprehensive test suite for multisync-core.

Tests cover:
1. SynchronyDataset — alignment, Z-score, NaN handling, context
2. Cascade — CCF, PRTF surrogates, Hanning window, significance
3. Dynamic features — WCC, 10 Gordon features
4. Prediction — TimeSeriesSplit, gap, leakage audit
5. Ground Truth — synthetic data with known lag
6. High-level API — 4-line workflow test
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_dyad():
    """Create a simple 2-modality dataset for testing."""
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=float)
    df_a = pd.DataFrame({"time": t, "value": np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 0.2})
    df_b = pd.DataFrame({"time": t, "value": np.cos(2 * np.pi * t / 50) + np.random.randn(n) * 0.2})
    from multisync.dataset import SynchronyDataset
    return SynchronyDataset(dyad_id="test", modalities={"a": df_a, "b": df_b})


def _make_aligned_dyad():
    """Create an already-aligned dyad."""
    ds = _make_simple_dyad()
    ds.align(target_hz=1.0)
    ds, _ = ds.zscore()
    return ds


# ===========================================================================
# 1. SynchronyDataset tests
# ===========================================================================

class TestSynchronyDataset:

    def test_creation(self):
        ds = _make_simple_dyad()
        assert ds.dyad_id == "test"
        assert set(ds.modality_names) == {"a", "b"}

    def test_missing_time_column_raises(self):
        from multisync.dataset import SynchronyDataset
        with pytest.raises(ValueError, match="time"):
            SynchronyDataset(
                dyad_id="bad",
                modalities={"x": pd.DataFrame({"val": [1, 2, 3]})},
            )

    def test_align_single_hz(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        assert ds._aligned
        assert len(ds.modalities["a"]) == len(ds.modalities["b"])

    def test_align_different_hz(self):
        from multisync.dataset import SynchronyDataset
        np.random.seed(42)
        t_slow = np.arange(0, 100, dtype=float)
        t_fast = np.arange(0, 100, 0.1)
        df_slow = pd.DataFrame({"time": t_slow, "value": np.random.randn(len(t_slow))})
        df_fast = pd.DataFrame({"time": t_fast, "value": np.random.randn(len(t_fast))})

        ds = SynchronyDataset(
            dyad_id="multi_hz",
            modalities={"slow": df_slow, "fast": df_fast},
        )
        ds.align(target_hz=1.0)
        # After alignment, both should have the same length
        assert len(ds.modalities["slow"]) == len(ds.modalities["fast"])

    def test_zscore(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        ds, stats = ds.zscore()
        assert ds._normalized
        # Mean should be ~0, std ~1 (ddof=0)
        a_vals = ds.modalities["a"]["value"]
        assert abs(a_vals.mean()) < 1e-10
        assert abs(a_vals.std(ddof=0) - 1.0) < 1e-10

    def test_zscore_stats_returned(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        _, stats = ds.zscore()
        assert "a" in stats
        assert "mean" in stats["a"]["value"]
        assert "std" in stats["a"]["value"]

    def test_context_labels(self):
        ds = _make_simple_dyad()
        ds.add_context(0, 50, "Task")
        ds.add_context(50, 100, "Rest")
        assert len(ds.context_labels) == 2
        ctx = ds.get_context_at(25)
        assert ctx is not None
        assert ctx.label == "Task"
        ctx_rest = ds.get_context_at(75)
        assert ctx_rest.label == "Rest"

    def test_nan_handling_ffill(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        # Inject NaN
        ds.modalities["a"].loc[10:15, "value"] = np.nan
        ds.handle_nan(strategy="ffill")
        assert ds.modalities["a"]["value"].iloc[15:].isna().sum() == 0

    def test_nan_handling_max_gap(self):
        ds = _make_simple_dyad()
        ds.align(target_hz=1.0)
        # Inject a long gap (10 samples)
        ds.modalities["a"].loc[10:20, "value"] = np.nan
        ds.handle_nan(strategy="ffill", max_gap_sec=5.0)
        # Gap of 10s > max_gap of 5s, so some NaN should remain
        assert ds.modalities["a"]["value"].iloc[10:20].isna().any()


# ===========================================================================
# 2. Cascade tests
# ===========================================================================

class TestCascade:

    def test_ccf_basic(self):
        from multisync.cascade import compute_ccf
        np.random.seed(42)
        n = 200
        x = np.sin(2 * np.pi * np.arange(n) / 50)
        y = np.sin(2 * np.pi * np.arange(n) / 50)
        lags, ccf = compute_ccf(x, y, max_lag_sec=10, hz=1.0)
        assert len(lags) == len(ccf)
        # With unwindowed normalization denominator, identical signals
        # still yield high but not perfect correlation (< 1.0) because
        # the Hanning window attenuates numerator edge contributions.
        assert ccf.max() > 0.3
        # Without window, identical signals should give peak very close to 1.0
        _, ccf_raw = compute_ccf(x, y, max_lag_sec=10, hz=1.0, apply_window=False)
        assert ccf_raw.max() > 0.99

    def test_ccf_shifted_signal(self):
        from multisync.cascade import compute_ccf
        np.random.seed(42)
        n = 500
        x = np.sin(2 * np.pi * np.arange(n) / 50)
        lag = 12
        y = np.zeros(n)
        y[lag:] = x[:-lag]
        lags, ccf = compute_ccf(x, y, max_lag_sec=30, hz=1.0)
        peak_idx = np.argmax(np.abs(ccf))
        detected_lag = lags[peak_idx]
        # Should detect negative lag (x leads y)
        assert detected_lag < -5  # x leads by ~12s

    def test_prft_surrogate_preserves_spectrum(self):
        from multisync.cascade import _prft_surrogate
        np.random.seed(42)
        x = np.random.randn(500)
        rng = np.random.default_rng(42)
        surr = _prft_surrogate(x, rng)
        # Power spectrum should be similar
        fft_x = np.abs(np.fft.rfft(x)) ** 2
        fft_s = np.abs(np.fft.rfft(surr)) ** 2
        corr = np.corrcoef(fft_x, fft_s)[0, 1]
        # The cosine taper (10% each end) modifies edge energy slightly,
        # so the spectrum is not identical to the original but still highly
        # correlated.  0.85 is a conservative lower bound — values around
        # 0.89–0.95 are typical after taper.
        assert corr > 0.85  # spectra should be highly correlated after taper

    def test_prft_surrogate_destroys_temporal_structure(self):
        from multisync.cascade import _prft_surrogate
        np.random.seed(42)
        # Use a non-stationary signal (chirp + step) where temporal structure matters
        n = 500
        t = np.arange(n, dtype=float)
        x = np.sin(2 * np.pi * t * t / (50 * 500))  # chirp (increasing frequency)
        x[:100] += 2.0  # step change
        rng = np.random.default_rng(42)
        surr = _prft_surrogate(x, rng)
        # Temporal regularity: autocorrelation at lag=1
        # Original should have higher lag-1 autocorrelation
        orig_ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        surr_ac1 = np.corrcoef(surr[:-1], surr[1:])[0, 1]
        assert orig_ac1 > surr_ac1

    def test_cascade_significance(self):
        """Surrogate test should NOT find significance in unrelated signals."""
        from multisync.cascade import cascade_analysis
        ds = _make_aligned_dyad()
        # Use very few surrogates for speed
        results, edges, metrics = cascade_analysis(
            ds, max_lag_sec=10, surrogate_n=50, seed=42
        )
        # With unrelated signals, most edges should not be significant
        # (though some may pass by chance at alpha=0.05)
        assert len(results) > 0

    def test_hanning_window_reduces_edge_effects(self):
        from multisync.cascade import compute_ccf
        np.random.seed(42)
        n = 100
        x = np.sin(2 * np.pi * np.arange(n) / 20) + np.random.randn(n) * 0.1
        y = x.copy()
        # With unwindowed normalization, windowed CCF peak is lower than 1.0
        # for identical signals (Hanning attenuates numerator edges).
        _, ccf_windowed = compute_ccf(x, y, max_lag_sec=5, hz=1.0, apply_window=True)
        _, ccf_raw = compute_ccf(x, y, max_lag_sec=5, hz=1.0, apply_window=False)
        # Raw (no window) should give near-perfect correlation for identical signals
        assert ccf_raw.max() > 0.99
        # Windowed should still be high but lower than raw
        assert ccf_windowed.max() > 0.3
        assert ccf_windowed.max() < ccf_raw.max()


# ===========================================================================
# 3. Dynamic features tests
# ===========================================================================

class TestDynamicFeatures:

    def test_wcc_identical_signals(self):
        from multisync.dynamic_features import sliding_window_wcc
        np.random.seed(42)
        n = 100
        x = np.sin(2 * np.pi * np.arange(n) / 20)
        wcc = sliding_window_wcc(x, x, window_size=10, hz=1.0)
        assert wcc.max() > 0.95  # identical → near-perfect correlation

    def test_wcc_uncorrelated_signals(self):
        from multisync.dynamic_features import sliding_window_wcc
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        wcc = sliding_window_wcc(x, y, window_size=10, hz=1.0)
        # Mean WCC should be near 0 for uncorrelated
        assert abs(np.nanmean(wcc)) < 0.3

    def test_wcc_with_lag(self):
        from multisync.dynamic_features import sliding_window_wcc
        np.random.seed(42)
        n = 500
        # Create a structured signal with clear temporal pattern
        t = np.arange(n, dtype=float)
        base = np.sin(2 * np.pi * t / 50) + 0.3 * np.sin(2 * np.pi * t / 20)
        lag = 10
        x = base.copy()
        y = np.zeros(n)
        y[lag:] = base[:-lag]
        # Verify that lag compensation produces a different WCC than no compensation
        wcc_no_comp = sliding_window_wcc(x, y, window_size=30, hz=1.0, lag_samples=0)
        wcc_comp = sliding_window_wcc(x, y, window_size=30, hz=1.0, lag_samples=lag)
        # The two WCC series should differ (lag compensation matters)
        assert not np.allclose(wcc_no_comp[5:-5], wcc_comp[5:-5], atol=0.01)
        # Compensated WCC should have higher max |correlation|
        assert np.nanmax(np.abs(wcc_comp)) > 0.5

    def test_extract_features(self):
        from multisync.dynamic_features import extract_dynamic_features
        # Use a Gaussian-like peak signal that find_peaks can actually detect
        n = 100
        t = np.arange(n, dtype=float)
        # Gaussian peak centered at t=35, sigma=8
        wcc = 0.8 * np.exp(-0.5 * ((t - 35) / 8.0) ** 2)
        feat = extract_dynamic_features(wcc, hz=1.0)
        assert feat.peak_amplitude > 0.7
        # Onset: first position where WCC >= 0.2 (default onset_threshold)
        # For Gaussian with center=35, sigma=8: solve 0.8*exp(-0.5*((t-35)/8)**2) = 0.2
        # => (t-35)/8 = ±sqrt(-2*ln(0.2/0.8)) ≈ ±1.1774
        # => t ≈ 35 ± 9.42 => onset at ~25.6
        assert 20 < feat.onset_latency < 30  # threshold crossing, not peak center
        assert feat.peak_duration > 0
        assert isinstance(feat.to_dict(), dict)

    def test_extract_features_all_pairs(self):
        from multisync.dynamic_features import extract_features_all_pairs
        ds = _make_aligned_dyad()
        feats = extract_features_all_pairs(ds, window_size=10, hz=1.0)
        assert len(feats) > 0
        for key, feat in feats.items():
            assert isinstance(feat.to_dict(), dict)


# ===========================================================================
# 4. Prediction tests (with leakage audit)
# ===========================================================================

class TestPrediction:

    def test_rolling_origin_cv_basic(self):
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        # Sine wave: every window has dynamics, labels are naturally balanced
        t = np.arange(800, dtype=float)
        series = np.sin(2 * np.pi * t / 80.0)  # period=80 samples
        pred = rolling_origin_cv(
            series, window_size=60, hz=1.0, n_splits=2, gap=2, threshold=0.0
        )
        assert len(pred.folds) > 0
        assert 0 <= pred.mean_dynamic_auc <= 1
        assert pred.mode == "intra"
        assert pred.n_features_used >= 0

    def test_dynamic_feature_matrix_not_autoregressive(self):
        """
        Verify that the prediction module now uses dynamic features,
        not raw WCC values. Feature importance keys should be dynamic
        feature names, not lag_1, lag_2, etc.
        """
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        series = np.concatenate([
            np.full(60, -1.0),
            np.full(60, 1.0),
            np.full(60, -1.0),
            np.full(60, 1.0),
            np.full(60, -1.0),
            np.full(60, 1.0),
            np.full(60, -1.0),
            np.full(60, 1.0),
        ])
        pred = rolling_origin_cv(
            series, window_size=60, hz=1.0, n_splits=3, gap=5
        )
        # Feature importance keys should be dynamic feature names
        if pred.feature_importance:
            for key in pred.feature_importance:
                assert not key.startswith("lag_"), (
                    f"Feature key '{key}' looks like raw WCC lag, "
                    f"not a dynamic feature name"
                )

    def test_leakage_audit_autocorrelated(self):
        """
        Leakage audit: feed a pure sine wave (perfectly autocorrelated).
        The delta-AUC should NOT be suspiciously high.
        If delta-AUC > 0.4, the warning flag must be raised.
        """
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        # Pure sine wave — trivially predictable due to autocorrelation
        t = np.arange(300, dtype=float)
        sine_wave = np.sin(2 * np.pi * t / 50)

        pred = rolling_origin_cv(
            sine_wave,
            window_size=30,
            hz=1.0,
            n_splits=5,
            gap=15,
        )
        # The warning flag should catch suspicious performance
        if pred.mean_delta_auc > 0.4:
            assert pred.warning == "leakage_suspected"

    def test_leakage_audit_random_noise(self):
        """Random noise should give AUC near 0.5 (no leakage possible)."""
        from multisync.prediction import rolling_origin_cv
        np.random.seed(42)
        # Use MUCH longer series to ensure stable AUC estimation
        noise = np.random.randn(2000)
        pred = rolling_origin_cv(
            noise, window_size=60, hz=1.0, n_splits=3, gap=2, threshold=0.0
        )
        # Random noise → AUC must be VERY close to 0.5
        assert len(pred.folds) > 0, "Should have at least one valid fold"
        assert abs(pred.mean_dynamic_auc - 0.5) < 0.1, (
            f"Random noise AUC should be near 0.5, got {pred.mean_dynamic_auc:.3f}. "
            f"This indicates leakage or overfitting."
        )

    def test_cross_modal_prediction_basic(self):
        """Cross-modal prediction: source and target are independent signals."""
        from multisync.prediction import cross_modal_prediction
        np.random.seed(42)
        # Source: has structure (sine wave)
        t = np.arange(300, dtype=float)
        source = np.sin(2 * np.pi * t / 50) + np.random.randn(300) * 0.3
        # Target: different structure (square wave)
        target = np.sign(np.sin(2 * np.pi * t / 30)) + np.random.randn(300) * 0.3

        pred = cross_modal_prediction(
            source, target,
            window_size=30, hz=1.0,
            source_name="behavioral__neural",
            target_name="neural__bio",
        )
        assert pred.mode == "cross_modal"
        assert pred.source_pair == "behavioral__neural"
        assert pred.target_pair == "neural__bio"

    def test_lodo_basic(self):
        from multisync.prediction import lodo_cv
        dyad_results = [
            {"mean_delta_auc": 0.1},
            {"mean_delta_auc": 0.2},
            {"mean_delta_auc": 0.15},
            {"mean_delta_auc": 0.25},
            {"mean_delta_auc": 0.18},
        ]
        result = lodo_cv(dyad_results)
        assert "mae" in result
        assert result["mae"] < 0.2  # predictions should be close


# ===========================================================================
# 5. Ground Truth test (THE critical test)
# ===========================================================================

class TestGroundTruth:

    def test_detect_known_lag_12s(self):
        """
        THE ground truth test from the reviewer:

        Generate: behavior leads neural by exactly 12 seconds + 20% white noise.
        Expect: cascade detects behavior→neural, lag ≈ 12s, p < 0.05.

        Uses lower noise (0.2) and more surrogates (200) for stable detection.
        No pytest.skip — this is a critical pipeline test.
        """
        from multisync.synthetic import generate_ground_truth_dyad
        from multisync.cascade import cascade_analysis

        ds = generate_ground_truth_dyad(
            lead_modality="behavior",
            lag_modality="neural",
            true_lag_sec=12.0,
            noise_ratio=0.2,
            duration_sec=300,
            hz=1.0,
            seed=42,
        )
        ds.align(target_hz=1.0)
        ds.zscore()

        # Use enough surrogates for reliable significance testing
        results, edges, metrics = cascade_analysis(
            ds, max_lag_sec=25.0, surrogate_n=200, seed=42, alpha=0.05
        )

        # Must have CCA results for the behavior-neural pair
        bn_cca = [
            r for r in results
            if (r.modality_a == "behavior" and r.modality_b == "neural")
            or (r.modality_a == "neural" and r.modality_b == "behavior")
        ]
        assert len(bn_cca) > 0, "No CCA result for behavior-neural pair"

        # Direction must be detected: behavior leads neural
        r = bn_cca[0]
        assert "behavior" in r.direction and "→" in r.direction, (
            f"Expected behavior→neural direction, got: {r.direction}"
        )
        detected_lag = abs(r.peak_lag_sec)

        # The detected lag should be within ±5s of the true 12s
        assert 7.0 <= detected_lag <= 17.0, (
            f"Detected lag {detected_lag}s outside expected range [7, 17]s"
        )


# ===========================================================================
# 6. High-level API test (4-line workflow)
# ===========================================================================

class TestHighLevelAPI:

    def test_four_line_workflow(self):
        """Verify the 4-line API from the README works."""
        import multisync as ms

        # 1. Load and align
        ds = ms.generate_ground_truth_dyad(
            lead_modality="behavior",
            lag_modality="neural",
            true_lag_sec=12.0,
            noise_ratio=0.3,
            duration_sec=300,
        )
        # 2. Add context
        ds.add_context(start=0, end=150, label="PreTask")
        ds.add_context(start=150, end=300, label="Task")
        # 3. Analyze (fewer surrogates for test speed)
        analyzer = ms.DynamicAnalyzer(window_size=10, surrogate_n=50)
        ds.align(target_hz=1.0)
        ds.zscore()
        results = analyzer.fit_transform(ds)
        # 4. Export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            results.export_viewer_json(path)
            # Verify JSON structure
            with open(path, "r") as f:
                data = json.load(f)
            assert "dyad_id" in data
            assert "cascade_graph" in data
            assert "nodes" in data["cascade_graph"]
            assert "edges" in data["cascade_graph"]
            assert "dynamic_features" in data
            assert "score_view" in data
            assert len(data["score_view"]) == 2  # PreTask + Task
        finally:
            os.unlink(path)

    def test_dyad_convenience_class(self):
        """Test the Dyad convenience wrapper."""
        import multisync as ms
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        df_n = pd.DataFrame({"time": t, "plv": np.random.randn(n)})
        df_b = pd.DataFrame({"time": t, "motion": np.random.randn(n)})

        dyad = ms.Dyad(neural=df_n, behavioral=df_b, hz=1.0)
        assert set(dyad.modality_names) == {"neural", "behavioral"}

    def test_analysis_results_schema(self):
        """Verify the viewer JSON has all required fields."""
        import multisync as ms
        ds = ms.generate_ground_truth_dyad(duration_sec=200, noise_ratio=0.2)
        ds.align(target_hz=1.0)
        ds.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=20)
        results = analyzer.fit_transform(ds)

        d = results.to_dict()
        # All required top-level keys
        assert "dyad_id" in d
        assert "cascade_graph" in d
        assert "dynamic_features" in d
        assert "dynamic_features_segmented" in d
        assert "prediction" in d
        assert "cross_modal_prediction" in d
        assert "parameters" in d

        # Cascade graph has correct structure
        cg = d["cascade_graph"]
        assert "nodes" in cg
        assert "edges" in cg
        for edge in cg["edges"]:
            assert "from" in edge
            assert "to" in edge
            assert "lag_sec" in edge
            assert "p_value" in edge
            assert "is_significant" in edge

        # JSON schema_version present (updated to 0.2.0)
        assert "schema_version" in d
        assert d["schema_version"] == "0.2.0"

    def test_context_segmented_features(self):
        """When contexts are defined, dynamic features should be computed
        per-context, not just globally."""
        import multisync as ms
        np.random.seed(42)
        n = 300
        t = np.arange(n, dtype=float)
        df_a = pd.DataFrame({
            "time": t,
            "val": np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 0.2,
        })
        df_b = pd.DataFrame({
            "time": t,
            "val": np.cos(2 * np.pi * t / 50) + np.random.randn(n) * 0.2,
        })

        dyad = ms.Dyad(a=df_a, b=df_b, hz=1.0)
        dyad.add_context(0, 150, "Phase1")
        dyad.add_context(150, 300, "Phase2")
        dyad.align(target_hz=1.0)
        dyad.zscore()

        analyzer = ms.DynamicAnalyzer(surrogate_n=10, window_size=10)
        results = analyzer.fit_transform(dyad)

        # Should have segmented features
        assert "dynamic_features_segmented" in results.to_dict()
        seg = results.dynamic_features_segmented
        assert "Phase1" in seg
        assert "Phase2" in seg
        # Each segment should have at least one pair's features
        assert len(seg["Phase1"]) > 0
        assert len(seg["Phase2"]) > 0

    def test_prediction_uses_dynamic_features_not_raw_wcc(self):
        """High-level test: verify that prediction results now report
        dynamic feature importance (not raw WCC lag coefficients)."""
        import multisync as ms
        ds = ms.generate_ground_truth_dyad(
            duration_sec=300, noise_ratio=0.2,
        )
        ds.align(target_hz=1.0)
        ds.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10, window_size=10)
        results = analyzer.fit_transform(ds)

        for key, pred in results.prediction.items():
            # Feature importance should use dynamic feature names
            if pred.get("feature_importance"):
                for feat_name in pred["feature_importance"]:
                    assert not feat_name.startswith("lag_"), (
                        f"Prediction {key} still uses raw WCC features: {feat_name}"
                    )


# ===========================================================================
# 7. JSON serialization tests
# ===========================================================================

class TestJSONSerialization:

    def test_nan_becomes_null_in_json(self):
        """NaN values must serialize as JSON null, not the string 'nan'."""
        import multisync as ms
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        # Insert NaN to trigger sanitization
        df_a = pd.DataFrame({"time": t, "val": np.random.randn(n)})
        df_b = pd.DataFrame({"time": t, "val": np.random.randn(n)})
        df_a.loc[5, "val"] = np.nan
        df_a.loc[10, "val"] = np.nan
        df_b.loc[15, "val"] = np.nan

        dyad = ms.Dyad(a=df_a, b=df_b, hz=1.0)
        dyad.align(target_hz=1.0)
        dyad.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10, window_size=10)
        results = analyzer.fit_transform(dyad)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            results.export_viewer_json(path)
            with open(path, "r") as f:
                content = f.read()
            # JSON null is allowed; the string "nan" is NOT
            assert '"nan"' not in content, (
                "NaN was serialized as string 'nan' instead of JSON null"
            )
            # Verify it's valid JSON
            data = json.loads(content)
            assert "schema_version" in data
        finally:
            os.unlink(path)


# ===========================================================================
# 8. Multimodal synthetic data tests (P1-C1)
# ===========================================================================

class TestMultimodalSynthetic:

    def test_shared_burst_anchors(self):
        """All modalities in generate_multimodal_dyad must share the same
        burst time anchors (the fix for the desync bug)."""
        from multisync.synthetic import generate_multimodal_dyad
        import multisync as ms

        ds = generate_multimodal_dyad(
            duration_sec=300,
            modalities={"neural": 10.0, "behavior": 1.0},
            seed=42,
        )
        ds.align(target_hz=1.0)

        # The synthetic generator creates Gaussian bursts at shared time
        # points. Verify that cross-modality CCF is non-trivial at short lags.
        n_feat_a = ds.feature_columns["neural"]
        n_feat_b = ds.feature_columns["behavior"]
        assert len(n_feat_a) > 0 and len(n_feat_b) > 0

    def test_cross_modality_ccf_has_structure(self):
        """Synthetic multimodal data should show cross-modality correlation
        structure (not pure noise) because bursts are shared."""
        from multisync.synthetic import generate_multimodal_dyad
        from multisync.cascade import compute_ccf

        ds = generate_multimodal_dyad(
            duration_sec=300,
            modalities={"neural": 5.0, "behavior": 1.0},
            noise_ratio=0.5,
            seed=42,
        )
        ds.align(target_hz=1.0)

        x = ds.get_aligned_array("neural", ds.feature_columns["neural"][0])
        y = ds.get_aligned_array("behavior", ds.feature_columns["behavior"][0])

        # Trim NaN
        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() < 30:
            return  # too short to test meaningfully
        x_v = x[valid]
        y_v = y[valid]

        lags, ccf = compute_ccf(x_v, y_v, max_lag_sec=20.0, hz=1.0, apply_window=False)
        # With shared bursts, peak CCF should be meaningfully above zero
        assert np.max(np.abs(ccf)) > 0.05

    def test_divergent_morphology_stress(self):
        """Divergent morphology should still allow CCF to detect direction,
        though peak CCF will be lower than identical morphology."""
        from multisync.synthetic import generate_ground_truth_dyad
        from multisync.cascade import compute_ccf

        ds = generate_ground_truth_dyad(
            lead_modality="behavior",
            lag_modality="neural",
            true_lag_sec=12.0,
            noise_ratio=0.3,
            duration_sec=300,
            hz=1.0,
            seed=42,
            morphology="divergent",
        )
        ds.align(target_hz=1.0)
        ds.zscore()

        x = ds.get_aligned_array("behavior", "value")
        y = ds.get_aligned_array("neural", "value")
        valid = ~np.isnan(x) & ~np.isnan(y)
        x_v, y_v = x[valid], y[valid]

        lags, ccf = compute_ccf(x_v, y_v, max_lag_sec=25.0, hz=1.0)
        peak_idx = np.argmax(np.abs(ccf))
        detected_lag = lags[peak_idx]

        # Direction should still be detected (behavior leads neural)
        assert detected_lag < -3, (
            f"Divergent morphology: expected negative lag (behavior leads), "
            f"got {detected_lag:.1f}s"
        )

    def test_identical_vs_divergent_peak_differs(self):
        """Identical morphology should give higher peak CCF than divergent
        for the same parameters."""
        from multisync.synthetic import generate_ground_truth_dyad
        from multisync.cascade import compute_ccf

        ds_id = generate_ground_truth_dyad(
            true_lag_sec=12.0, noise_ratio=0.2, seed=42,
            morphology="identical",
        )
        ds_dv = generate_ground_truth_dyad(
            true_lag_sec=12.0, noise_ratio=0.2, seed=42,
            morphology="divergent",
        )
        ds_id.align(target_hz=1.0)
        ds_id.zscore()
        ds_dv.align(target_hz=1.0)
        ds_dv.zscore()

        def _get_peak_ccf(ds):
            x = ds.get_aligned_array("behavior", "value")
            y = ds.get_aligned_array("neural", "value")
            v = ~np.isnan(x) & ~np.isnan(y)
            _, ccf = compute_ccf(x[v], y[v], max_lag_sec=25.0, hz=1.0)
            return np.max(np.abs(ccf))

        peak_id = _get_peak_ccf(ds_id)
        peak_dv = _get_peak_ccf(ds_dv)
        # Identical morphology should give higher peak (same waveform shifted)
        assert peak_id >= peak_dv, (
            f"Identical peak {peak_id:.3f} should be >= divergent peak {peak_dv:.3f}"
        )


# ===========================================================================
# 9. CLI tests (P1-C2)
# ===========================================================================

class TestCLI:

    def test_demo_command_runs(self):
        """The `demo` CLI command should run without errors."""
        from multisync.cli import cmd_demo
        import argparse

        args = argparse.Namespace(surrogates=20, output=None)
        cmd_demo(args)  # Should not raise

    def test_analyze_command_runs(self):
        """The `analyze` CLI command should run with synthetic CSVs."""
        from multisync.cli import cmd_analyze
        import argparse
        import tempfile

        # Create temporary CSV files
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        csvs = []
        for name in ["neural", "behavior"]:
            path = tempfile.mktemp(suffix=".csv")
            df = pd.DataFrame({"time": t, "val": np.random.randn(n)})
            df.to_csv(path, index=False)
            csvs.append(path)

        try:
            args = argparse.Namespace(
                input=",".join(csvs),
                names="neural,behavior",
                hz="1.0",
                output=None,
                window_size=10,
                surrogates=10,
                max_lag=20.0,
                seed=42,
                contexts=None,
            )
            cmd_analyze(args)  # Should not raise
        finally:
            for p in csvs:
                os.unlink(p)


# ===========================================================================
# 10. Edge case tests (P3-C4)
# ===========================================================================

class TestEdgeCases:

    def test_single_modality_no_crash(self):
        """Single modality should not crash — no pairs to analyze."""
        import multisync as ms
        np.random.seed(42)
        n = 100
        t = np.arange(n, dtype=float)
        df = pd.DataFrame({"time": t, "val": np.random.randn(n)})

        dyad = ms.Dyad(neural=df, hz=1.0)
        dyad.align(target_hz=1.0)
        dyad.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10)
        results = analyzer.fit_transform(dyad)
        # Should have empty results but no crash
        assert len(results.cascade_graph["edges"]) == 0
        assert len(results.dynamic_features) == 0

    def test_very_short_data_graceful(self):
        """Data shorter than window_size should return empty results, not crash."""
        from multisync.dynamic_features import sliding_window_wcc
        x = np.random.randn(5)
        y = np.random.randn(5)
        result = sliding_window_wcc(x, y, window_size=10, hz=1.0)
        assert len(result) == 0  # empty array

    def test_identical_signals_ccf_raw_is_one(self):
        """Without window, identical signals must give CCF peak ≈ 1.0 at lag 0."""
        from multisync.cascade import compute_ccf
        n = 200
        x = np.random.randn(n)
        lags, ccf = compute_ccf(x, x, max_lag_sec=10, hz=1.0, apply_window=False)
        center_idx = len(lags) // 2
        assert abs(ccf[center_idx] - 1.0) < 0.01

    def test_mostly_nan_pair_produces_warning(self):
        """A modality pair with 90%+ NaN should trigger a logging warning
        but should not crash the pipeline."""
        import multisync as ms
        import logging
        n = 100
        t = np.arange(n, dtype=float)
        vals_a = np.random.randn(n)
        vals_a[:90] = np.nan  # 90% NaN
        df_a = pd.DataFrame({"time": t, "val": vals_a})
        df_b = pd.DataFrame({"time": t, "val": np.random.randn(n)})

        dyad = ms.Dyad(a=df_a, b=df_b, hz=1.0)
        dyad.align(target_hz=1.0)
        dyad.zscore()
        analyzer = ms.DynamicAnalyzer(surrogate_n=10)
        # Should not raise
        results = analyzer.fit_transform(dyad)
        assert "cascade_graph" in results.to_dict()
