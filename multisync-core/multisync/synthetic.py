"""
Synthetic data generator — Ground Truth validation data.

Creates controlled multi-modal dyadic data with known temporal relationships
(e.g., "behavior leads neural by exactly 12 seconds") plus configurable
noise, for algorithm validation.

Two morphology modes:
- "identical" (default): lead and lag use the same waveform shape, shifted.
  Best for unit tests — CCF should reliably detect the lag.
- "divergent": lead uses Gaussian bursts, lag uses exponential decays.
  Best for stress tests — validates that CCF handles different waveform
  morphologies (closer to real cross-modal data where neural theta
  suppression ≠ behavioral nodding, but event timing is linked).

Usage:
    from multisync.synthetic import generate_ground_truth_dyad
    ds = generate_ground_truth_dyad(
        lead_modality="behavior",
        lag_modality="neural",
        true_lag_sec=12.0,
        noise_ratio=0.3,
        duration_sec=300,
        hz=1.0,
    )
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset import SynchronyDataset


def generate_ground_truth_dyad(
    lead_modality: str = "behavior",
    lag_modality: str = "neural",
    true_lag_sec: float = 12.0,
    noise_ratio: float = 0.3,
    duration_sec: float = 300.0,
    hz: float = 1.0,
    seed: int = 42,
    n_bursts: int = 5,
    burst_sigma: float = 3.0,
    gap_prob: float = 0.02,
    morphology: Literal["identical", "divergent"] = "identical",
) -> SynchronyDataset:
    """
    Generate a synthetic dyad with a known lead-lag relationship.

    Mechanism (morphology="identical", default):
    1. Create a base signal (sum of Gaussian bursts at random times).
    2. The lead modality gets the base signal directly.
    3. The lag modality gets the base signal shifted by ``true_lag_sec``.
    4. White noise is added at ``noise_ratio * signal_std``.
    5. Optional NaN gaps simulate device dropout.

    Mechanism (morphology="divergent"):
    Same event anchors, but different waveform shapes:
    - Lead modality: Gaussian bursts (sharp onset).
    - Lag modality: Asymmetric exponential decays (long tail).
    This simulates real cross-modal data where "neural theta suppression"
    and "behavioral nodding" have different morphologies but linked timing.

    Parameters
    ----------
    lead_modality, lag_modality : str
        Names for the two modalities.
    true_lag_sec : float
        How many seconds the lead modality precedes the lag modality.
    noise_ratio : float
        Noise amplitude as fraction of signal standard deviation (0.3 = 30%).
    duration_sec : float
        Total recording length in seconds.
    hz : float
        Sampling rate.
    seed : int
        Random seed for reproducibility.
    n_bursts : int
        Number of events in the base signal.
    burst_sigma : float
        Width (seconds) of each Gaussian burst.
    gap_prob : float
        Probability per sample of inserting a NaN (simulates dropout).
    morphology : str
        "identical" — both modalities use the same waveform (unit tests).
        "divergent" — different waveforms, same event anchors (stress test).

    Returns
    -------
    SynchronyDataset
        Ready for alignment and analysis.  The true relationship is:
        ``lead_modality`` precedes ``lag_modality`` by ``true_lag_sec``.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * hz)
    t = np.arange(n) / hz
    lag_samples = int(true_lag_sec * hz)

    # 1. Event anchors (shared between modalities)
    burst_times = rng.uniform(20, duration_sec - 20 - true_lag_sec, size=n_bursts)

    if morphology == "identical":
        # Both modalities: Gaussian bursts (same shape)
        base_lead = np.zeros(n)
        for bt in burst_times:
            base_lead += np.exp(-0.5 * ((t - bt) / burst_sigma) ** 2)
        base_lead += 0.3 * np.sin(2 * np.pi * t / 60.0)

        signal_std = float(np.std(base_lead))
        lead = base_lead + rng.normal(0, noise_ratio * signal_std, size=n)
        lag = np.zeros(n)
        lag[lag_samples:] = base_lead[:-lag_samples] + rng.normal(
            0, noise_ratio * signal_std, size=n - lag_samples
        )

    elif morphology == "divergent":
        # Lead: Gaussian bursts (sharp onset, symmetric)
        lead_signal = np.zeros(n)
        for bt in burst_times:
            lead_signal += np.exp(-0.5 * ((t - bt) / burst_sigma) ** 2)
        lead_signal += 0.3 * np.sin(2 * np.pi * t / 60.0)

        # Lag: Asymmetric exponential decay (long rise, slow decay)
        # This models physiological responses that have different temporal
        # dynamics from behavioral events (e.g., neural response to a
        # behavioral cue has a delayed onset and prolonged decay).
        lag_signal = np.zeros(n)
        tau_rise = burst_sigma * 0.5   # fast rise
        tau_decay = burst_sigma * 2.5  # slow decay
        for bt in burst_times:
            dt = t - bt - true_lag_sec * 0.3  # slight inherent delay
            # Asymmetric: fast rise, slow decay
            lag_signal += np.where(
                dt >= 0,
                np.exp(-dt / tau_decay),
                np.exp(dt / tau_rise),
            )
        lag_signal += 0.2 * np.sin(2 * np.pi * t / 45.0 + 1.5)

        signal_std_lead = float(np.std(lead_signal))
        signal_std_lag = float(np.std(lag_signal))

        lead = lead_signal + rng.normal(0, noise_ratio * signal_std_lead, size=n)
        lag = lag_signal + rng.normal(0, noise_ratio * signal_std_lag, size=n)

    else:
        raise ValueError(f"Unknown morphology: {morphology}. Use 'identical' or 'divergent'.")

    # Add NaN gaps (device dropout)
    if gap_prob > 0:
        nan_mask_lead = rng.random(n) < gap_prob
        nan_mask_lag = rng.random(n) < gap_prob
        lead[nan_mask_lead] = np.nan
        lag[nan_mask_lag] = np.nan

    # Create DataFrames
    df_lead = pd.DataFrame({"time": t, "value": lead})
    df_lag = pd.DataFrame({"time": t, "value": lag})

    ds = SynchronyDataset(
        dyad_id=f"synthetic_lag{true_lag_sec}s_noise{noise_ratio}_{morphology}",
        modalities={lead_modality: df_lead, lag_modality: df_lag},
    )

    # Store ground truth as metadata
    ds._ground_truth = {
        "lead": lead_modality,
        "lag": lag_modality,
        "true_lag_sec": true_lag_sec,
        "noise_ratio": noise_ratio,
        "n_bursts": n_bursts,
        "morphology": morphology,
    }

    return ds


def generate_multimodal_dyad(
    duration_sec: float = 300.0,
    hz: float = 1.0,
    seed: int = 42,
    modalities: Optional[Dict[str, float]] = None,
    noise_ratio: float = 0.3,
) -> SynchronyDataset:
    """
    Generate a synthetic dyad with 3-4 modalities at different Hz.

    Parameters
    ----------
    modalities : dict or None
        {modality_name: original_hz}.  Default:
        {"neural": 1.0, "behavior": 10.0, "bio": 4.0}
    """
    if modalities is None:
        modalities = {"neural": 1.0, "behavior": 10.0, "bio": 4.0}

    rng = np.random.default_rng(seed)

    # Shared burst times — all modalities use the SAME temporal anchors
    # so that cross-modality synchrony is genuinely present in Ground Truth.
    # Each modality then applies a fixed offset to simulate lead-lag.
    shared_bursts = rng.uniform(20, duration_sec - 40, size=5)

    dataframes = {}
    for name, orig_hz in modalities.items():
        n = int(duration_sec * orig_hz)
        t = np.arange(n) / orig_hz

        signal = np.zeros(n)
        for bt in shared_bursts:
            # Offset each modality
            offset = {"neural": 0, "behavior": -5, "bio": -3}.get(name, 0)
            signal += np.exp(-0.5 * ((t - bt - offset) / 3.0) ** 2)

        signal += 0.2 * np.sin(2 * np.pi * t / 45.0 + hash(name) % 10)
        signal += rng.normal(0, noise_ratio * np.std(signal), size=n)

        dataframes[name] = pd.DataFrame({"time": t, "value": signal})

    return SynchronyDataset(
        dyad_id="synthetic_multimodal",
        modalities=dataframes,
    )
