/**
 * Dynamic Feature Extraction Module
 * 
 * Extracts time-dynamic features from epoch-level synchrony timeseries
 * for characterizing the process of synchrony emergence (not just its level).
 * 
 * Music analogy: these features describe HOW the notes become a symphony,
 * not just whether they're loud or quiet.
 * 
 * Features:
 * 1. Onset Latency — time until first significant synchrony
 * 2. Build-up Rate — slope from baseline to peak (z-score/epoch)
 * 3. Peak Amplitude — maximum synchrony value
 * 4. Time to Peak — epoch at which peak occurs
 * 5. Maintenance Duration — time spent above threshold
 * 6. Stability (CV) — coefficient of variation above threshold
 * 7. Breakdown Count — number of drops below threshold
 * 8. Recovery Rate — average epochs to recover after breakdown
 * 9. Cascade Onset Delay — delay between first and last modality onset
 * 10. Entrainment Entropy — regularity of the synchrony rhythm
 */

export interface DynamicFeatureSet {
  modality: string;
  // Temporal features
  onsetLatencyEpochs: number | null;
  onsetLatencySec: number | null;
  timeToPeakEpochs: number;
  timeToPeakSec: number;
  // Amplitude features
  peakValue: number;
  meanValue: number;
  minValue: number;
  dynamicRange: number; // peak - min
  // Process features
  buildUpRate: number | null; // z-score units per epoch
  maintenanceDurationEpochs: number | null;
  maintenanceDurationSec: number | null;
  maintenanceStabilityCV: number | null; // CV during maintenance phase
  breakdownCount: number;
  avgRecoveryEpochs: number | null;
  // Rhythm features
  entrainmentEntropy: number; // Shannon entropy of discretized signal
}

export interface CrossModalityFeatureSet {
  cascadeOnsetDelayEpochs: number | null;
  cascadeOnsetDelaySec: number | null;
  leaderModality: string | null;
  followerModality: string | null;
  cascadePattern: string; // e.g., "behavioral → bio → neural"
  multimodalRMS: number; // RMS of all modalities at each epoch
  multimodalCoherenceEpochs: number; // epochs where >2 modalities above threshold
  fullChordProportion: number; // proportion of epochs with ALL modalities above threshold
}

export interface DynamicFeatureReport {
  perModality: DynamicFeatureSet[];
  crossModality: CrossModalityFeatureSet;
  epochMs: number;
  thresholdSigma: number;
  summary: string;
}

/**
 * Extract dynamic features for a single modality timeseries.
 */
export function extractDynamicFeatures(
  values: number[],
  epochMs: number,
  thresholdSigma: number = 0.5
): DynamicFeatureSet {
  const n = values.length;
  if (n === 0) {
    return emptyFeatureSet("unknown", epochMs);
  }

  const mean = values.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  const peakValue = Math.max(...values);
  const minValue = Math.min(...values);
  const peakEpoch = values.indexOf(peakValue);
  const dynamicRange = peakValue - minValue;

  // Onset detection: first epoch where value >= threshold (z-score)
  let onsetEpoch: number | null = null;
  for (let i = 0; i < n - 1; i++) {
    if (values[i] >= thresholdSigma && values[i + 1] >= thresholdSigma) {
      onsetEpoch = i;
      break;
    }
  }

  // Build-up rate: slope from onset to peak
  let buildUpRate: number | null = null;
  if (onsetEpoch !== null && peakEpoch > onsetEpoch) {
    const span = peakEpoch - onsetEpoch;
    if (span > 0) {
      buildUpRate = (values[peakEpoch] - values[onsetEpoch]) / span;
    }
  }

  // Maintenance duration: consecutive epochs at or above threshold after onset
  let maintenanceDurationEpochs: number | null = null;
  let maintenanceStabilityCV: number | null = null;
  if (onsetEpoch !== null) {
    let duration = 0;
    const maintainedValues: number[] = [];
    for (let i = onsetEpoch; i < n; i++) {
      if (values[i] >= thresholdSigma) {
        duration++;
        maintainedValues.push(values[i]);
      } else {
        break; // first drop below threshold
      }
    }
    maintenanceDurationEpochs = duration > 0 ? duration : null;

    if (maintainedValues.length > 2) {
      const mMean = maintainedValues.reduce((a, b) => a + b, 0) / maintainedValues.length;
      const mStd = Math.sqrt(maintainedValues.reduce((a, b) => a + (b - mMean) ** 2, 0) / maintainedValues.length);
      maintenanceStabilityCV = mMean !== 0 ? mStd / Math.abs(mMean) : null;
    }
  }

  // Breakdown & Recovery analysis
  let breakdownCount = 0;
  const recoveryEpochs: number[] = [];
  if (onsetEpoch !== null) {
    let inMaintenance = false;
    let breakdownStart = 0;
    for (let i = onsetEpoch; i < n; i++) {
      const aboveThreshold = values[i] >= thresholdSigma;
      if (inMaintenance && !aboveThreshold) {
        breakdownCount++;
        breakdownStart = i;
        inMaintenance = false;
      } else if (!inMaintenance && aboveThreshold) {
        if (breakdownStart > 0) {
          recoveryEpochs.push(i - breakdownStart);
        }
        inMaintenance = true;
      }
    }
  }

  const avgRecoveryEpochs = recoveryEpochs.length > 0
    ? recoveryEpochs.reduce((a, b) => a + b, 0) / recoveryEpochs.length
    : null;

  // Entrainment entropy: discretize signal and compute Shannon entropy
  // Higher entropy = more irregular rhythm (less "musical")
  const entrainmentEntropy = computeEntrainmentEntropy(values, 10);

  return {
    modality: "",
    onsetLatencyEpochs: onsetEpoch,
    onsetLatencySec: onsetEpoch !== null ? parseFloat(((onsetEpoch * epochMs) / 1000).toFixed(1)) : null,
    timeToPeakEpochs: peakEpoch,
    timeToPeakSec: parseFloat(((peakEpoch * epochMs) / 1000).toFixed(1)),
    peakValue: parseFloat(peakValue.toFixed(4)),
    meanValue: parseFloat(mean.toFixed(4)),
    minValue: parseFloat(minValue.toFixed(4)),
    dynamicRange: parseFloat(dynamicRange.toFixed(4)),
    buildUpRate: buildUpRate !== null ? parseFloat(buildUpRate.toFixed(4)) : null,
    maintenanceDurationEpochs,
    maintenanceDurationSec: maintenanceDurationEpochs !== null
      ? parseFloat(((maintenanceDurationEpochs * epochMs) / 1000).toFixed(1))
      : null,
    maintenanceStabilityCV: maintenanceStabilityCV !== null ? parseFloat(maintenanceStabilityCV.toFixed(4)) : null,
    breakdownCount,
    avgRecoveryEpochs: avgRecoveryEpochs !== null ? parseFloat(avgRecoveryEpochs.toFixed(1)) : null,
    entrainmentEntropy: parseFloat(entrainmentEntropy.toFixed(4)),
  };
}

/**
 * Compute Shannon entropy of a discretized signal.
 * Measures the "regularity" of the synchrony rhythm.
 * Low entropy = regular, predictable pattern (strong entrainment).
 * High entropy = irregular, noisy pattern (weak entrainment).
 */
function computeEntrainmentEntropy(values: number[], bins: number): number {
  if (values.length === 0) return 0;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  if (range === 0) return 0;

  const binWidth = range / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const bin = Math.min(bins - 1, Math.floor((v - min) / binWidth));
    counts[bin]++;
  }

  let entropy = 0;
  for (const count of counts) {
    if (count > 0) {
      const p = count / values.length;
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

function emptyFeatureSet(modality: string, epochMs: number): DynamicFeatureSet {
  return {
    modality,
    onsetLatencyEpochs: null,
    onsetLatencySec: null,
    timeToPeakEpochs: 0,
    timeToPeakSec: 0,
    peakValue: 0,
    meanValue: 0,
    minValue: 0,
    dynamicRange: 0,
    buildUpRate: null,
    maintenanceDurationEpochs: null,
    maintenanceDurationSec: null,
    maintenanceStabilityCV: null,
    breakdownCount: 0,
    avgRecoveryEpochs: null,
    entrainmentEntropy: 0,
  };
}

/**
 * Extract cross-modality features from multiple modality timeseries.
 */
export function extractCrossModalityFeatures(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  thresholdSigma: number = 0.5
): CrossModalityFeatureSet {
  const mods = Object.keys(modalityTimeseries);
  const n = Math.min(...Object.values(modalityTimeseries).map((v) => v.length));

  if (n === 0 || mods.length === 0) {
    return {
      cascadeOnsetDelayEpochs: null,
      cascadeOnsetDelaySec: null,
      leaderModality: null,
      followerModality: null,
      cascadePattern: "",
      multimodalRMS: 0,
      multimodalCoherenceEpochs: 0,
      fullChordProportion: 0,
    };
  }

  // Find onset for each modality
  const onsets: { mod: string; epoch: number }[] = [];
  for (const [mod, vals] of Object.entries(modalityTimeseries)) {
    const v = vals.slice(0, n);
    let onset: number | null = null;
    for (let i = 0; i < v.length - 1; i++) {
      if (v[i] >= thresholdSigma && v[i + 1] >= thresholdSigma) {
        onset = i;
        break;
      }
    }
    if (onset !== null) onsets.push({ mod, epoch: onset });
  }

  // Sort by onset time
  onsets.sort((a, b) => a.epoch - b.epoch);

  const leaderModality = onsets.length > 0 ? onsets[0].mod : null;
  const followerModality = onsets.length > 1 ? onsets[onsets.length - 1].mod : null;

  let cascadeOnsetDelayEpochs: number | null = null;
  if (onsets.length >= 2) {
    cascadeOnsetDelayEpochs = onsets[onsets.length - 1].epoch - onsets[0].epoch;
  }

  const cascadePattern = onsets.map((o) => o.mod).join(" → ");

  // Multimodal coherence metrics
  let coherenceEpochs = 0;
  let fullChordEpochs = 0;
  let rmsSum = 0;

  for (let i = 0; i < n; i++) {
    const epochValues = mods.map((m) => modalityTimeseries[m][i] ?? 0);
    const aboveThreshold = epochValues.filter((v) => v >= thresholdSigma).length;
    if (aboveThreshold >= 2) coherenceEpochs++;
    if (aboveThreshold === mods.length) fullChordEpochs++;

    const rms = Math.sqrt(epochValues.reduce((sum, v) => sum + v ** 2, 0) / epochValues.length);
    rmsSum += rms;
  }

  return {
    cascadeOnsetDelayEpochs,
    cascadeOnsetDelaySec: cascadeOnsetDelayEpochs !== null
      ? parseFloat(((cascadeOnsetDelayEpochs * epochMs) / 1000).toFixed(1))
      : null,
    leaderModality,
    followerModality,
    cascadePattern,
    multimodalRMS: parseFloat((rmsSum / n).toFixed(4)),
    multimodalCoherenceEpochs: coherenceEpochs,
    fullChordProportion: n > 0 ? parseFloat((fullChordEpochs / n).toFixed(4)) : 0,
  };
}

/**
 * Run complete dynamic feature extraction.
 */
export function runDynamicFeatureExtraction(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  thresholdSigma: number = 0.5
): DynamicFeatureReport {
  // Per-modality features
  const perModality: DynamicFeatureSet[] = [];
  for (const [mod, values] of Object.entries(modalityTimeseries)) {
    const features = extractDynamicFeatures(values, epochMs, thresholdSigma);
    features.modality = mod;
    perModality.push(features);
  }

  // Cross-modality features
  const crossModality = extractCrossModalityFeatures(modalityTimeseries, epochMs, thresholdSigma);

  // Build summary
  const summaryParts: string[] = [];
  if (crossModality.leaderModality) {
    summaryParts.push(
      `Cascade leader: ${crossModality.leaderModality} (onset at ${perModality.find(f => f.modality === crossModality.leaderModality)?.onsetLatencySec ?? "N/A"}s).`
    );
  }
  if (crossModality.cascadeOnsetDelaySec !== null) {
    summaryParts.push(
      `Cascade span: ${crossModality.cascadeOnsetDelaySec}s (${crossModality.cascadePattern}).`
    );
  }
  summaryParts.push(
    `Full-chord proportion: ${(crossModality.fullChordProportion * 100).toFixed(1)}% of epochs with all modalities above ${thresholdSigma}σ.`
  );

  // Highlight interesting features
  const fastBuilder = perModality.find(f => f.buildUpRate !== null && f.buildUpRate > 0.1);
  if (fastBuilder) {
    summaryParts.push(
      `${fastBuilder.modality} shows rapid build-up (${fastBuilder.buildUpRate?.toFixed(3)} z/epoch).`
    );
  }

  const fragile = perModality.find(f => f.breakdownCount >= 2);
  if (fragile) {
    summaryParts.push(
      `${fragile.modality} is fragile: ${fragile.breakdownCount} breakdowns, avg recovery ${fragile.avgRecoveryEpochs ?? "N/A"} epochs.`
    );
  }

  return {
    perModality,
    crossModality,
    epochMs,
    thresholdSigma,
    summary: summaryParts.join(" "),
  };
}
