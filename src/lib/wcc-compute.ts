/**
 * WCC Computation Module
 * 
 * Implements Windowed Cross-Correlation with proper statistical preprocessing:
 * - Per-window linear detrending (Moulder et al., 2018)
 * - Multiple normalization strategies
 * - Epoch aggregation for multi-timescale alignment
 * - Direction-aware WCC results (positive/negative/mixed synchrony)
 */

export type StreamData = { t: number; p1: number; p2: number };

export type NormalizationMethod = "zscore" | "zscore_baseline" | "minmax";

export interface NormalizationConfig {
  method: NormalizationMethod;
  baselineEndMs?: number;
}

/** Direction-aware WCC result per window */
export interface WCCWindowResult {
  peakCorrelation: number;    // signed peak correlation (preserves direction)
  peakAbsCorrelation: number; // absolute value for backward compat
  lagAtPeak: number;          // lag in samples at peak
  lagAtPeakMs: number;        // lag in ms at peak
  direction: "positive" | "negative";
}

/**
 * Linear detrend: removes best-fit line from a signal segment.
 */
function linearDetrend(signal: number[]): number[] {
  const n = signal.length;
  if (n < 2) return signal;

  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += signal[i];
    sumXY += i * signal[i];
    sumX2 += i * i;
  }
  const denom = n * sumX2 - sumX * sumX;
  if (Math.abs(denom) < 1e-12) return signal;

  const b = (n * sumXY - sumX * sumY) / denom;
  const a = (sumY - b * sumX) / n;

  return signal.map((val, i) => val - (a + b * i));
}

/**
 * Compute Windowed Cross-Correlation with direction-aware output.
 * 
 * Returns an array of WCCWindowResult with signed peak correlation,
 * lag at peak, and direction classification.
 */
export function computeWCCDirectional(
  data: StreamData[],
  windowMs: number,
  lagMs: number,
  sampleRateHz: number,
  detrend: boolean = true
): WCCWindowResult[] {
  const windowSamples = Math.max(1, Math.round((windowMs / 1000) * sampleRateHz));
  const results: WCCWindowResult[] = [];
  const increment = Math.max(1, Math.floor(windowSamples / 2));
  const msPerSample = 1000 / sampleRateHz;

  for (let i = 0; i < data.length - windowSamples; i += increment) {
    let w1 = data.slice(i, i + windowSamples).map((d) => d.p1);
    let w2 = data.slice(i, i + windowSamples).map((d) => d.p2);

    if (detrend) {
      w1 = linearDetrend(w1);
      w2 = linearDetrend(w2);
    }

    const m1 = w1.reduce((a, b) => a + b, 0) / w1.length;
    const m2 = w2.reduce((a, b) => a + b, 0) / w2.length;
    const s1 = Math.sqrt(w1.reduce((a, b) => a + (b - m1) ** 2, 0) / w1.length);
    const s2 = Math.sqrt(w2.reduce((a, b) => a + (b - m2) ** 2, 0) / w2.length);

    if (s1 === 0 || s2 === 0) {
      results.push({ peakCorrelation: 0, peakAbsCorrelation: 0, lagAtPeak: 0, lagAtPeakMs: 0, direction: "positive" });
      continue;
    }

    let bestCorr = 0;
    let bestAbsCorr = 0;
    let bestLag = 0;
    const lagSamples = Math.round((lagMs / 1000) * sampleRateHz);

    for (let lag = -lagSamples; lag <= lagSamples; lag++) {
      let sum = 0, count = 0;
      for (let j = 0; j < windowSamples; j++) {
        const j2 = j + lag;
        if (j2 >= 0 && j2 < windowSamples) {
          sum += ((w1[j] - m1) / s1) * ((w2[j2] - m2) / s2);
          count++;
        }
      }
      const corr = count > 0 ? sum / count : 0;
      if (Math.abs(corr) > bestAbsCorr) {
        bestAbsCorr = Math.abs(corr);
        bestCorr = corr;
        bestLag = lag;
      }
    }

    results.push({
      peakCorrelation: bestCorr,
      peakAbsCorrelation: bestAbsCorr,
      lagAtPeak: bestLag,
      lagAtPeakMs: parseFloat((bestLag * msPerSample).toFixed(1)),
      direction: bestCorr >= 0 ? "positive" : "negative",
    });
  }
  return results;
}

/**
 * Backward-compatible: returns absolute peak correlations (legacy behavior).
 */
export function computeWCC(
  data: StreamData[],
  windowMs: number,
  lagMs: number,
  sampleRateHz: number,
  detrend: boolean = true
): number[] {
  return computeWCCDirectional(data, windowMs, lagMs, sampleRateHz, detrend)
    .map(r => r.peakAbsCorrelation);
}

/**
 * Session-global z-scoring.
 */
export function zScore(values: number[]): number[] {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  if (std === 0) return values.map(() => 0);
  return values.map((v) => (v - mean) / std);
}

/**
 * Baseline z-scoring.
 */
export function zScoreBaseline(values: number[], baselineCount: number): number[] {
  if (baselineCount < 3 || baselineCount >= values.length) {
    return zScore(values);
  }

  const baselineSlice = values.slice(0, baselineCount);
  const mean = baselineSlice.reduce((a, b) => a + b, 0) / baselineSlice.length;
  const std = Math.sqrt(baselineSlice.reduce((a, b) => a + (b - mean) ** 2, 0) / baselineSlice.length);
  if (std === 0) return zScore(values);
  return values.map((v) => (v - mean) / std);
}

/**
 * Min-max normalization to [0, 1].
 */
export function minMaxNormalize(values: number[]): number[] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return values.map(() => 0.5);
  return values.map((v) => (v - min) / (max - min));
}

/**
 * Apply normalization based on config.
 */
export function normalize(
  values: number[],
  config: NormalizationConfig,
  wccWindowsPerSec?: number
): number[] {
  switch (config.method) {
    case "zscore_baseline": {
      const baselineSec = (config.baselineEndMs || 60000) / 1000;
      const baselineCount = Math.round(baselineSec * (wccWindowsPerSec || 1));
      return zScoreBaseline(values, baselineCount);
    }
    case "minmax":
      return minMaxNormalize(values);
    case "zscore":
    default:
      return zScore(values);
  }
}

/**
 * Epoch aggregation: average values within each epoch.
 */
export function epochAggregate(values: number[], epochSamples: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < values.length; i += epochSamples) {
    const chunk = values.slice(i, i + epochSamples);
    result.push(chunk.reduce((a, b) => a + b, 0) / chunk.length);
  }
  return result;
}

/**
 * Epoch-aggregate directional WCC results, computing per-epoch:
 * - mean signed correlation
 * - mean absolute correlation
 * - fraction negative (for direction classification)
 */
export interface EpochDirectionalResult {
  meanSigned: number;
  meanAbs: number;
  fractionNegative: number;
  direction: "positive" | "negative" | "mixed";
  meanLagMs: number;
}

export function epochAggregateDirectional(
  windows: WCCWindowResult[],
  epochSamples: number
): EpochDirectionalResult[] {
  const result: EpochDirectionalResult[] = [];
  for (let i = 0; i < windows.length; i += epochSamples) {
    const chunk = windows.slice(i, i + epochSamples);
    if (chunk.length === 0) continue;

    const meanSigned = chunk.reduce((a, w) => a + w.peakCorrelation, 0) / chunk.length;
    const meanAbs = chunk.reduce((a, w) => a + w.peakAbsCorrelation, 0) / chunk.length;
    const negCount = chunk.filter(w => w.direction === "negative").length;
    const fractionNeg = negCount / chunk.length;
    const meanLag = chunk.reduce((a, w) => a + w.lagAtPeakMs, 0) / chunk.length;

    let direction: "positive" | "negative" | "mixed";
    if (fractionNeg > 0.7) direction = "negative";
    else if (fractionNeg < 0.3) direction = "positive";
    else direction = "mixed";

    result.push({
      meanSigned: parseFloat(meanSigned.toFixed(4)),
      meanAbs: parseFloat(meanAbs.toFixed(4)),
      fractionNegative: parseFloat(fractionNeg.toFixed(2)),
      direction,
      meanLagMs: parseFloat(meanLag.toFixed(1)),
    });
  }
  return result;
}
