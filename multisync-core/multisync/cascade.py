"""
Cascade analysis — Cross-Correlation Function with rigorous surrogate testing.

Key design decisions (Nature Methods reviewer requirements):
- CCF computed via numpy correlate (vectorized, no Python for-loops).
- Edge-effect mitigation via Hanning (tapered cosine) window.
- Surrogate testing via Phase-Randomized Fourier Transform (PRTF), which
  destroys temporal structure while preserving the power spectrum and
  amplitude distribution.  Far more rigorous than circular shift.
- p-value: fraction of surrogates whose |peak CCF| >= observed |peak CCF|.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import correlate as sp_correlate, detrend as sp_detrend

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CCAResult:
    """Result of a single cross-correlation analysis between two modalities."""
    modality_a: str
    modality_b: str
    feature_a: str = ""
    feature_b: str = ""
    lags_sec: np.ndarray = field(default_factory=lambda: np.array([]))
    ccf_values: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_lag_sec: float = 0.0
    peak_ccf: float = 0.0
    direction: str = ""
    # Significance
    is_significant: bool = False
    p_value: float = 1.0
    p_value_corrected: float = 1.0  # BH FDR corrected p-value
    surrogate_n: int = 0
    null_peak_ccf: Optional[np.ndarray] = None  # surrogates' peak CCFs
    # Diagnostics (runtime warnings)
    diagnostics: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Convert arrays to lists for JSON serialization
        lags_sec_list = self.lags_sec.tolist() if len(self.lags_sec) > 0 else []
        ccf_list = self.ccf_values.tolist() if len(self.ccf_values) > 0 else []
        null_peaks = self.null_peak_ccf.tolist() if self.null_peak_ccf is not None and len(self.null_peak_ccf) > 0 else None
        return {
            "modality_a": self.modality_a,
            "modality_b": self.modality_b,
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "lags_sec": lags_sec_list,
            "ccf_values": ccf_list,
            "peak_lag_sec": float(self.peak_lag_sec),
            "peak_ccf": float(self.peak_ccf),
            "direction": self.direction,
            "is_significant": self.is_significant,
            "p_value": float(self.p_value),
            "p_value_corrected": float(self.p_value_corrected),
            "surrogate_n": self.surrogate_n,
            "null_peak_ccf": null_peaks,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CCAResult":
        """Deserialize from a dict (inverse of to_dict)."""
        lags = np.array(d.get("lags_sec", []), dtype=float)
        ccf = np.array(d.get("ccf_values", []), dtype=float)
        null_peaks = np.array(d["null_peak_ccf"], dtype=float) if d.get("null_peak_ccf") is not None else None
        return cls(
            modality_a=d.get("modality_a", ""),
            modality_b=d.get("modality_b", ""),
            feature_a=d.get("feature_a", ""),
            feature_b=d.get("feature_b", ""),
            lags_sec=lags,
            ccf_values=ccf,
            peak_lag_sec=float(d.get("peak_lag_sec", 0.0)),
            peak_ccf=float(d.get("peak_ccf", 0.0)),
            direction=d.get("direction", ""),
            is_significant=bool(d.get("is_significant", False)),
            p_value=float(d.get("p_value", 1.0)),
            p_value_corrected=float(d.get("p_value_corrected", d.get("p_value", 1.0))),
            surrogate_n=int(d.get("surrogate_n", 0)),
            null_peak_ccf=null_peaks,
            diagnostics=d.get("diagnostics", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CCAResult":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


@dataclass
class CascadeEdge:
    """A directed edge in the cascade graph (Viewer JSON ready)."""
    source: str
    target: str
    lag_sec: float
    ccf_value: float
    p_value: float
    is_significant: bool
    polarity: str = "positive"  # "positive" (excitatory) or "negative" (inhibitory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "lag_sec": float(self.lag_sec),
            "ccf_value": float(self.ccf_value),
            "p_value": float(self.p_value),
            "is_significant": self.is_significant,
            "polarity": self.polarity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CascadeEdge":
        """Deserialize from a dict (inverse of to_dict)."""
        return cls(
            source=d.get("source", ""),
            target=d.get("target", ""),
            lag_sec=float(d.get("lag_sec", 0.0)),
            ccf_value=float(d.get("ccf_value", 0.0)),
            p_value=float(d.get("p_value", 1.0)),
            is_significant=bool(d.get("is_significant", False)),
            polarity=d.get("polarity", "positive"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CascadeEdge":
        """Deserialize from a JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Surrogate generation — Phase-Randomized Fourier Transform (PRTF)
# ---------------------------------------------------------------------------

def _prft_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
    taper_fraction: float = 0.1,
) -> np.ndarray:
    """
    Generate a Phase-Randomized Fourier Transform surrogate.

    Steps:
    1. Apply a short cosine taper to the signal edges to suppress spectral
       leakage from discontinuous endpoints (edge jump artifact).
       The taper covers *taper_fraction* of the signal on each side
       (default 10%), leaving the central 80% untouched.  This is a split
       cosine bell — NOT a full Hanning window — so the bulk of the power
       spectrum is preserved.  Applying a full window here would distort
       the power spectrum that PRTF is supposed to maintain.
    2. FFT of the tapered signal.
    3. Randomize phases uniformly in [-pi, pi], except keep DC and Nyquist
       phases fixed (preserve mean and amplitude distribution).
    4. Inverse FFT → surrogate with same power spectrum, destroyed temporal
       structure.

    Parameters
    ----------
    x : 1-D array
        Original time series.
    rng : numpy Generator
        Random number generator (for reproducibility).
    taper_fraction : float
        Fraction of signal length to taper at each end (default 0.1 = 10%).

    Returns
    -------
    surrogate : 1-D array, same length as x.
    """
    n = len(x)
    if n < 4:
        return x.copy()

    # --- Step 1: edge taper to eliminate spectral leakage ---
    # Build a split cosine bell: ramp up on the first taper_len samples,
    # flat 1.0 in the middle, ramp down on the last taper_len samples.
    # This is mathematically a Tukey window with alpha = 2 * taper_fraction.
    taper_len = max(1, int(np.floor(n * taper_fraction)))
    taper = np.ones(n, dtype=float)
    # Left ramp: 0 → 1 over taper_len samples using a half cosine
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len) / taper_len))
    taper[:taper_len] = ramp
    taper[-taper_len:] = ramp[::-1]
    x_tapered = x * taper

    # --- Step 2: FFT ---
    X = fft(x_tapered)

    # --- Step 3: Randomize phases (preserve DC at index 0 and Nyquist) ---
    phases = np.angle(X)
    random_phases = rng.uniform(-np.pi, np.pi, size=n)

    # Keep DC and Nyquist (if even-length)
    random_phases[0] = phases[0]
    if n % 2 == 0:
        random_phases[n // 2] = phases[n // 2]

    # Enforce Hermitian symmetry: X[k] = conj(X[n-k])
    X_new = np.zeros_like(X)
    half = n // 2 + 1  # include Nyquist for even n
    for k in range(half):
        magnitude = np.abs(X[k])
        new_phase = random_phases[k]
        X_new[k] = magnitude * np.exp(1j * new_phase)
        if k > 0 and k < n - k:
            # Mirror frequency (skip when k == n-k, i.e., Nyquist for even n)
            X_new[n - k] = np.conj(X_new[k])

    # --- Step 4: IFFT ---
    surrogate = np.real(ifft(X_new))
    return surrogate


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _hanning_window(n: int) -> np.ndarray:
    """Periodic Hanning window (scipy.signal.windows.hann equivalent)."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


# ---------------------------------------------------------------------------
# CCF computation (vectorized)
# ---------------------------------------------------------------------------

def compute_ccf(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_sec: float,
    hz: float,
    apply_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation function via numpy correlate.

    Parameters
    ----------
    x, y : 1-D arrays
        Input time series (same length, same sampling rate).
    max_lag_sec : float
        Maximum lag to compute, in seconds.
    hz : float
        Sampling rate in Hz.
    apply_window : bool
        If True, apply Hanning window before CCF to reduce edge effects.

    Returns
    -------
    lags_sec : 1-D array
        Lag values in seconds.
    ccf : 1-D array
        Normalized CCF values (Pearson-like, range ≈ [-1, 1]).

    Notes
    -----
    **Nonlinear drift caveat**: This function uses `scipy.signal.detrend(type="linear")`
    to remove linear slow drift. However, if the signal contains **nonlinear drift**
    (e.g., exponential, polynomial, or U-shaped slow components), linear detrending
    will **fail to remove it completely**, potentially creating **U-shaped pseudo-oscillations**
    in the CCF. For signals with visible nonlinear baseline wander, consider:
    
    - Using `scipy.signal.detrend(type="constant")` (demean only) + manual nonlinear removal
    - Applying a high-pass filter before CCF
    - Visual inspection of detrended signals
    
    See: Issue #3 "线性去趋势导致的U型伪振荡骗局" in project notes.
    """
    n = len(x)
    assert len(y) == n, f"x and y must have same length: {n} vs {len(y)}"

    max_lag_samples = int(np.floor(max_lag_sec * hz))

    if n < 2 * max_lag_samples + 1:
        raise ValueError(
            f"Series length ({n}) too short for max_lag ({max_lag_samples} samples). "
            f"Need at least {2 * max_lag_samples + 1}."
        )

    # Detrend: remove linear trend (slow drift) then demean.
    # Using scipy.signal.detrend instead of simple x - x.mean() because
    # biological signals (e.g, EDA) often have global slow drift (gradual
    # sweating across a 5-min session).  Shared linear trends dominate CCF
    # and push peak_idx to max lag boundaries, creating false causality.
    # detrend(type="linear") fits and subtracts the least-squares line,
    # removing slow drift while preserving local fluctuations.
    x_detrended = sp_detrend(x)
    y_detrended = sp_detrend(y)
    # After detrend, mean is ~0, but re-demean for numerical safety
    x_demean = x_detrended - x_detrended.mean()
    y_demean = y_detrended - y_detrended.mean()

    # Apply Hanning window for edge-effect mitigation
    if apply_window:
        win = _hanning_window(n)
        x_w = x_demean * win
        y_w = y_demean * win
    else:
        x_w = x_demean
        y_w = y_demean

    # Normalized cross-correlation via FFT convolution — O(n log n).
    # scipy.signal.correlate with method="fft" uses the FFT convolution theorem,
    # which reduces complexity from O(n²) (direct convolution) to O(n log n).
    # This is critical when surrogate_n is large (e.g., 500 surrogates × n²
    # operations would stall on signals with n > 3000 samples).
    nccf = sp_correlate(x_w, y_w, mode="full", method="fft")

    # Normalize to Pearson correlation using UNWINDOWED denominator.
    # The standard Pearson normalization uses the variance of the raw
    # demeaned signal, not the window-weighted variance.  This ensures
    # CCF values stay in [-1, 1] — the window only tapers edge
    # contributions to the numerator (cross-product sum), not the scale.
    norm_x = np.sqrt(np.sum(x_demean ** 2))
    norm_y = np.sqrt(np.sum(y_demean ** 2))
    if norm_x > 0 and norm_y > 0:
        nccf = nccf / (norm_x * norm_y)

    # Extract the lag range: negative lags (x leads) to positive lags (y leads)
    center = n - 1
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    valid = center + lags
    ccf = nccf[valid]

    lags_sec = lags / hz

    return lags_sec, ccf


# ---------------------------------------------------------------------------
# Full cascade analysis with surrogate testing
# ---------------------------------------------------------------------------

def cascade_analysis(
    dataset: "SynchronyDataset",  # noqa: F821  forward ref
    max_lag_sec: float = 30.0,
    surrogate_n: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    apply_window: bool = True,
) -> Tuple[List[CCAResult], List[CascadeEdge], Dict[str, Dict[str, Any]]]:
    """
    Compute pairwise CCF across all modalities with PRTF surrogate testing.

    Parameters
    ----------
    dataset : SynchronyDataset
        Must be aligned and normalized.
    max_lag_sec : float
        Maximum cross-correlation lag in seconds.
    surrogate_n : int
        Number of PRTF surrogates for significance testing.
    alpha : float
        Significance threshold.  ``is_significant`` is True when
        p < alpha.
    seed : int
        Random seed for reproducibility.
    apply_window : bool
        Apply Hanning window before CCF.

    Returns
    -------
    cca_results : list of CCAResult
        Detailed results per modality pair (with BH-corrected p-values).
    edges : list of CascadeEdge
        Viewer-ready directed cascade edges (only significant ones after FDR).
    metrics : dict
        Lightweight graph metrics (in_degree, out_degree, driver_score,
        is_hub, is_follower) per modality.  Computed without networkx.
    """
    if not dataset._aligned:
        raise ValueError("Dataset must be aligned before cascade analysis.")

    hz = dataset.target_hz
    feat_cols = dataset.feature_columns
    modality_names = dataset.modality_names
    rng = np.random.default_rng(seed)

    cca_results: List[CCAResult] = []
    edges: List[CascadeEdge] = []
    raw_pvals: List[float] = []  # collect raw p-values for BH correction

    for i, name_a in enumerate(modality_names):
        for name_b in modality_names[i + 1:]:
            for col_a in feat_cols[name_a]:
                for col_b in feat_cols[name_b]:
                    x = dataset.get_aligned_array(name_a, col_a)
                    y = dataset.get_aligned_array(name_b, col_b)
                    if x is None or y is None:
                        continue

                    # Trim leading and trailing NaN only — never slice internal gaps.
                    either_nan = np.isnan(x) | np.isnan(y)
                    if either_nan.sum() == len(x):
                        logger.debug(
                            "Skipping %s/%s x %s/%s: entirely NaN",
                            name_a, col_a, name_b, col_b,
                        )
                        continue  # entirely NaN
                    # Find first/last valid index to trim edges
                    first_valid = int(np.argmax(~either_nan))
                    last_valid = len(either_nan) - 1 - int(np.argmax(~either_nan[::-1]))
                    x_trim = x[first_valid : last_valid + 1].copy()
                    y_trim = y[first_valid : last_valid + 1].copy()
                    if len(x_trim) < 20:
                        logger.debug(
                            "Skipping %s/%s x %s/%s: segment too short (%d < 20)",
                            name_a, col_a, name_b, col_b, len(x_trim),
                        )
                        continue
                    # Fill internal NaN with local mean instead of 0.
                    # Zero-filling creates false synchrony when both signals
                    # have NaN at the same positions (0 vs 0 = "perfect sync").
                    # Local mean is a more conservative imputation.
                    nan_ratio = np.isnan(x_trim).sum() / len(x_trim)
                    if nan_ratio > 0.1:
                        logger.warning(
                            "High NaN ratio (%.1f%%) in %s/%s x %s/%s, "
                            "results may be unreliable",
                            nan_ratio * 100, name_a, col_a, name_b, col_b,
                        )
                    x_mean = np.nanmean(x_trim)
                    y_mean = np.nanmean(y_trim)
                    x_clean = np.where(np.isnan(x_trim), x_mean if not np.isnan(x_mean) else 0.0, x_trim)
                    y_clean = np.where(np.isnan(y_trim), y_mean if not np.isnan(y_mean) else 0.0, y_trim)

                    # --- Nonlinear drift detection (strict heuristic) ---
                    # Only trigger when linear detrending FAILED to remove
                    # the dominant trend: if residual variance >95% of original,
                    # the drift is essentially nonlinear and CCF may contain
                    # U-shaped pseudo-oscillations.
                    # Threshold is strict (95%) to avoid Alert Fatigue:
                    # most EDA/fNIRS signals have mild nonlinear components
                    # that do NOT invalidate CCF results.
                    diagnostics = []  # local diagnostics for this pair
                    for sig, label in [(x_clean, f"{name_a}/{col_a}"), (y_clean, f"{name_b}/{col_b}")]:
                        sig_valid = sig[~np.isnan(sig)]
                        if len(sig_valid) < 30:
                            continue
                        var_orig = np.var(sig_valid)
                        var_res = np.var(sp_detrend(sig_valid))
                        # Trigger only if residual variance > 95% of original
                        # (linear detrending removed <5% of variance)
                        if var_orig > 0 and var_res > 0.95 * var_orig:
                            diagnostics.append({
                                "type": "warning",
                                "message": (
                                    f"Severe nonlinear drift detected in {label}. "
                                    f"Linear detrending removed <5% of signal variance. "
                                    f"CCF results may contain spurious oscillations. "
                                    f"Recommendation: apply high-pass filter (e.g., 0.01Hz) "
                                    f"in preprocessing (MNE/NeuroKit) before importing."
                                )
                            })

                    # Cap max_lag_sec to the mathematically feasible range for
                    # this particular segment.  CCF needs n >= 2*max_lag + 1,
                    # so max_lag <= (n-1)//2.  Silently cap instead of crashing.
                    n_seg = len(x_clean)
                    feasible_lag_samples = (n_seg - 1) // 2
                    feasible_lag_sec = feasible_lag_samples / hz
                    effective_max_lag = min(max_lag_sec, feasible_lag_sec)
                    if effective_max_lag < max_lag_sec:
                        logger.info(
                            "Capped max_lag from %.1fs to %.1fs for "
                            "%s/%s x %s/%s (segment length %d)",
                            max_lag_sec, effective_max_lag,
                            name_a, col_a, name_b, col_b, n_seg,
                        )

                    # Degrees-of-freedom guard: CCF + Surrogate testing on very
                    # short segments produces noise peaks, not real synchrony.
                    # Require n >= 3 * effective_max_lag_samples + 1 so that
                    # there are at least 3 full lag-windows of data.  If not,
                    # skip this pair entirely rather than emit noisy results.
                    min_required = 3 * int(np.floor(effective_max_lag * hz)) + 1
                    if n_seg < min_required:
                        logger.warning(
                            "Skipping %s/%s x %s/%s: segment (%d samples) is "
                            "too short for reliable CCF+Surrogate analysis "
                            "(need >= %d = 3×max_lag_samples+1). "
                            "Results would be statistical noise.",
                            name_a, col_a, name_b, col_b, n_seg, min_required,
                        )
                        continue

                    # Compute observed CCF
                    lags_sec, ccf_vals = compute_ccf(
                        x_clean, y_clean, effective_max_lag, hz, apply_window
                    )

                    # Find peak
                    abs_ccf = np.abs(ccf_vals)
                    peak_idx = int(np.argmax(abs_ccf))
                    peak_lag = float(lags_sec[peak_idx])
                    peak_val = float(ccf_vals[peak_idx])

                    # Determine direction
                    if peak_lag < 0:
                        direction = f"{name_a}→{name_b}"
                        source, target = name_a, name_b
                        peak_lag = abs(peak_lag)
                    elif peak_lag > 0:
                        direction = f"{name_b}→{name_a}"
                        source, target = name_b, name_a
                        peak_lag = abs(peak_lag)
                    else:
                        direction = "synchronous"
                        source, target = name_a, name_b

                    # --- Surrogate testing (PRTF) ---
                    null_peaks = np.empty(surrogate_n)
                    for s in range(surrogate_n):
                        x_surr = _prft_surrogate(x_clean, rng)
                        y_surr = _prft_surrogate(y_clean, rng)
                        _, ccf_s = compute_ccf(
                            x_surr, y_surr, effective_max_lag, hz, apply_window
                        )
                        null_peaks[s] = np.max(np.abs(ccf_s))

                    # p-value: Phipson & Smyth (2010) unbiased permutation
                    # p-value estimator.  The observed statistic is treated as
                    # one draw from the null distribution, so the minimum
                    # possible p is 1 / (surrogate_n + 1) rather than 0.
                    #
                    # Formula: p = (1 + #{surrogates >= |observed|}) / (N + 1)
                    #
                    # Why not np.mean(null > obs)?
                    #   - With N=500, a perfectly extreme signal gives p=0.0,
                    #     which is statistically non-standard and rejected by
                    #     most journals.  The +1 correction ensures p ∈ (0, 1].
                    # Why >= instead of >?
                    #   - Strict > is anti-conservative on discrete null
                    #     distributions; >= gives correct coverage (Phipson &
                    #     Smyth, Stat Appl Genet Mol Biol, 2010).
                    surrogate_n_actual = len(null_peaks)
                    p_val = float(
                        (1.0 + np.sum(null_peaks >= abs(peak_val)))
                        / (surrogate_n_actual + 1.0)
                    )
                    is_sig = p_val < alpha
                    raw_pvals.append(p_val)

                    result = CCAResult(
                        modality_a=name_a,
                        modality_b=name_b,
                        feature_a=col_a,
                        feature_b=col_b,
                        lags_sec=lags_sec,
                        ccf_values=ccf_vals,
                        peak_lag_sec=peak_lag,
                        peak_ccf=peak_val,
                        direction=direction,
                        is_significant=is_sig,
                        p_value=p_val,
                        surrogate_n=surrogate_n,
                        null_peak_ccf=null_peaks,
                        diagnostics=diagnostics,
                    )
                    cca_results.append(result)

                    if is_sig and direction != "synchronous":
                        edges.append(CascadeEdge(
                            source=source,
                            target=target,
                            lag_sec=peak_lag,
                            ccf_value=peak_val,
                            p_value=p_val,
                            is_significant=True,
                            polarity="positive" if peak_val >= 0 else "negative",
                        ))

        # --- Post-hoc: Benjamini-Hochberg FDR correction ---
        # Raw p-values from pairwise surrogate tests are subject to the
        # multiple-comparisons problem.  With m modality pairs, the family-wise
        # error rate at alpha=0.05 approaches 1 - (0.95)^m (e.g, 40% for
        # m=10, ~90% for m=45).  Apply BH correction to control FDR.
        if len(raw_pvals) > 0:
            corrected = _bh_fdr_correct(raw_pvals, q=alpha)
            # Update cca_results with corrected p-values
            # Also rebuild edges based on corrected significance
            edges = []
            for idx, result in enumerate(cca_results):
                result.p_value_corrected = corrected[idx]
                result.is_significant = corrected[idx] < alpha
                # Update p_value to corrected (for downstream consumption)
                result.p_value = corrected[idx]

                # Rebuild edges with corrected significance
                if result.is_significant and result.direction != "synchronous":
                    # Determine source/target from direction string
                    if "→" in result.direction:
                        src, tgt = result.direction.split("→")
                    else:
                        src = result.modality_a
                        tgt = result.modality_b
                    edges.append(CascadeEdge(
                        source=src,
                        target=tgt,
                        lag_sec=result.peak_lag_sec,
                        ccf_value=result.peak_ccf,
                        p_value=result.p_value_corrected,
                        is_significant=result.is_significant,
                        polarity="positive" if result.peak_ccf >= 0 else "negative",
                    ))

        # --- Compute lightweight graph metrics (no networkx) ---
        metrics = compute_cascade_metrics(edges, alpha)

        return cca_results, edges, metrics

    # --- Compute lightweight graph metrics (no networkx) ---
        metrics = compute_cascade_metrics(edges, alpha)

        return cca_results, edges, metrics


# ---------------------------------------------------------------------------
# Multiple Comparisons Correction — Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------

def _bh_fdr_correct(p_values: List[float], q: float = 0.05) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.

    Returns adjusted p-values (q-values).  A test is significant if
    adjusted_p <= q.

    Parameters
    ----------
    p_values : list of float
        Raw p-values (uncorrected).
    q : float
        Target FDR level (default 0.05).

    Returns
    -------
    adjusted : list of float
        Adjusted p-values, same order as input.
        adjusted[i] <= q  →  reject H_0 (significant).

    Notes
    -----
    **PRDS assumption**: BH procedure assumes **Positive Regression Dependency
    on a Subset (PRDS)** for strict FDR control. If p-values are strongly
    **negatively correlated** (e.g., one significant result makes others less
    likely to be significant), the actual FDR may **exceed q**.

    In MultiSync, CCF p-values from **overlapping time windows** may violate
    PRDS. Consider:
    
    - Using **independent tests only** (e.g., separate dyads, separate epochs)
    - Applying **permutation-based FDR** (more robust to dependence)
    - Reporting **raw p-values + correction method** for transparency

    See: Issue #4 "FDR的关联依赖性（PRDS假设）隐患" in project notes.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    indexed = [(i, p) for i, p in enumerate(p_values)]
    indexed_sorted = sorted(indexed, key=lambda x: x[1])

    # Compute adjusted p-values (BH step-up)
    # adjusted_p_(j) = min_{k >= j} (m * p_(k) / k)
    # Then enforce monotonicity: adj_p_(j) <= adj_p_(j+1)
    raw_adj = [0.0] * m
    for j in range(1, m + 1):
        p_j = indexed_sorted[j - 1][1]
        raw_adj[j - 1] = min(1.0, p_j * m / j)

    # Enforce non-decreasing order (adjusted p-values can only increase)
    for j in range(m - 2, -1, -1):
        raw_adj[j] = min(raw_adj[j], raw_adj[j + 1])

    # Map back to original order
    adjusted = [0.0] * m
    for j in range(m):
        orig_idx = indexed_sorted[j][0]
        adjusted[orig_idx] = raw_adj[j]

    return adjusted


# ---------------------------------------------------------------------------
# Lightweight Graph Metrics (no networkx dependency)
# ---------------------------------------------------------------------------

def compute_cascade_metrics(
    edges: List[CascadeEdge],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute lightweight graph metrics from cascade edges.

    Uses only collections.Counter — no networkx needed.
    Computes:
    - in_degree: how many edges point TO this modality (driven by others)
    - out_degree: how many edges originate FROM this modality (drives others)
    - driver_score: out_degree - in_degree (positive = driver, negative = follower)

    Parameters
    ----------
    edges : list of CascadeEdge
        Directed edges (only significant ones).
    alpha : float
        Significance threshold (for labeling; already filtered by caller).

    Returns
    -------
    metrics : dict
        {modality_name: {"in_degree": int, "out_degree": int,
                         "driver_score": int, "is_hub": bool}}
        is_hub is True if out_degree >= 2 (drives 2+ others).
    """
    from collections import Counter

    out_degrees = Counter(e.source for e in edges if e.is_significant)
    in_degrees = Counter(e.target for e in edges if e.is_significant)

    all_modalities = set(out_degrees.keys()) | set(in_degrees.keys())
    metrics: Dict[str, Dict[str, Any]] = {}

    for mod in all_modalities:
        out_d = out_degrees.get(mod, 0)
        in_d = in_degrees.get(mod, 0)
        driver = out_d - in_d
        metrics[mod] = {
            "in_degree": in_d,
            "out_degree": out_d,
            "driver_score": driver,
            "is_hub": out_d >= 2,       # drives 2+ others → hub/driver
            "is_follower": in_d >= 2,   # driven by 2+ others → follower
        }

    return metrics





# Re-export for convenience
from .dataset import SynchronyDataset  # noqa: E402, F401
