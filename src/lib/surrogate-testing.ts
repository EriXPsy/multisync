/**
 * Surrogate Testing Module
 * 
 * Implements permutation/surrogate methods to establish statistical significance
 * of observed synchrony values. Without surrogate baselines, observed WCC values
 * cannot be distinguished from chance co-fluctuation.
 * 
 * Methods:
 * 1. **Phase-shuffled surrogates** — Randomize phase while preserving amplitude
 *    spectrum and autocorrelation structure (Theiler et al., 1992).
 * 2. **Pseudo-pair surrogates** — Pair participant A's signal with participant B
 *    from a different time window (Moulder et al., 2018).
 * 
 * The observed WCC is compared against a distribution of surrogate WCCs
 * to compute a percentile rank (p-value).
 */

import { computeWCC, type StreamData } from "./wcc-compute";

export interface SurrogateResult {
  streamName: string;
  modality: string;
  observedMeanWCC: number;
  surrogateMean: number;
  surrogateSD: number;
  percentileRank: number;   // 0-100: percentile of observed in surrogate distribution
  pValue: number;           // 1 - percentileRank/100
  significant: boolean;     // p < 0.05
  nSurrogates: number;
  method: "phase_shuffle" | "pseudo_pair";
}

/**
 * Phase-shuffle surrogate: randomizes the phase of a signal's FFT
 * while preserving its power spectrum. This destroys synchrony
 * while keeping individual signal properties (autocorrelation, variance).
 * 
 * Simplified implementation using random circular shift (which preserves
 * autocorrelation exactly) — adequate for screening, computationally cheap.
 */
function circularShift(signal: number[]): number[] {
  const n = signal.length;
  if (n < 2) return [...signal];
  const shift = Math.floor(Math.random() * (n - 1)) + 1;
  return [...signal.slice(shift), ...signal.slice(0, shift)];
}

/**
 * Generate pseudo-pair surrogate data by shifting one participant's
 * timeseries by a large random offset relative to the other.
 * This breaks temporal coupling while preserving individual dynamics.
 */
function pseudoPairSurrogate(data: StreamData[]): StreamData[] {
  const n = data.length;
  if (n < 4) return data;
  
  // Shift p2 by a random amount (at least 25% of signal length)
  const minShift = Math.floor(n * 0.25);
  const maxShift = Math.floor(n * 0.75);
  const shift = minShift + Math.floor(Math.random() * (maxShift - minShift));
  
  const p2Values = data.map(d => d.p2);
  const shiftedP2 = circularShift(p2Values);
  // Apply additional random shift
  const finalP2 = [...shiftedP2.slice(shift % n), ...shiftedP2.slice(0, shift % n)];
  
  return data.map((d, i) => ({
    t: d.t,
    p1: d.p1,
    p2: finalP2[i] ?? d.p2,
  }));
}

/**
 * Compute mean absolute WCC for a dataset.
 */
function meanAbsWCC(wccValues: number[]): number {
  if (wccValues.length === 0) return 0;
  return wccValues.reduce((a, b) => a + Math.abs(b), 0) / wccValues.length;
}

/**
 * Run surrogate testing for a single data stream.
 * 
 * @param data - Raw dyadic timeseries
 * @param windowMs - WCC window size
 * @param lagMs - WCC max lag
 * @param sampleRateHz - Sampling rate
 * @param nSurrogates - Number of surrogates (default 200)
 * @param method - Surrogate method
 * @returns SurrogateResult with significance assessment
 */
export function runSurrogateTest(
  data: StreamData[],
  windowMs: number,
  lagMs: number,
  sampleRateHz: number,
  streamName: string,
  modality: string,
  nSurrogates: number = 200,
  method: "phase_shuffle" | "pseudo_pair" = "pseudo_pair"
): SurrogateResult {
  // Compute observed WCC
  const observedWCC = computeWCC(data, windowMs, lagMs, sampleRateHz, true);
  const observedMean = meanAbsWCC(observedWCC);

  // Generate surrogate distribution
  const surrogateMeans: number[] = [];

  for (let i = 0; i < nSurrogates; i++) {
    let surrogateData: StreamData[];

    if (method === "pseudo_pair") {
      surrogateData = pseudoPairSurrogate(data);
    } else {
      // Phase shuffle: circularly shift p2
      const p2Values = data.map(d => d.p2);
      const shuffled = circularShift(p2Values);
      surrogateData = data.map((d, j) => ({ t: d.t, p1: d.p1, p2: shuffled[j] }));
    }

    const surrogateWCC = computeWCC(surrogateData, windowMs, lagMs, sampleRateHz, true);
    surrogateMeans.push(meanAbsWCC(surrogateWCC));
  }

  // Compute statistics
  const surrogateMean = surrogateMeans.reduce((a, b) => a + b, 0) / surrogateMeans.length;
  const surrogateSD = Math.sqrt(
    surrogateMeans.reduce((a, b) => a + (b - surrogateMean) ** 2, 0) / surrogateMeans.length
  );

  // Percentile rank: what fraction of surrogates is the observed value above?
  const nBelow = surrogateMeans.filter(s => s < observedMean).length;
  const percentileRank = (nBelow / nSurrogates) * 100;
  const pValue = 1 - percentileRank / 100;

  return {
    streamName,
    modality,
    observedMeanWCC: parseFloat(observedMean.toFixed(4)),
    surrogateMean: parseFloat(surrogateMean.toFixed(4)),
    surrogateSD: parseFloat(surrogateSD.toFixed(4)),
    percentileRank: parseFloat(percentileRank.toFixed(1)),
    pValue: parseFloat(pValue.toFixed(4)),
    significant: pValue < 0.05,
    nSurrogates,
    method,
  };
}

/**
 * Run surrogate tests for multiple streams.
 * Uses reduced surrogate count (100) for speed in browser context.
 */
export function runSurrogateTestBatch(
  streams: Array<{
    data: StreamData[];
    name: string;
    modality: string;
    sampleRateHz: number;
  }>,
  windowMs: number = 5000,
  lagMs: number = 2000,
  nSurrogates: number = 100
): SurrogateResult[] {
  return streams.map(stream =>
    runSurrogateTest(
      stream.data,
      windowMs,
      lagMs,
      stream.sampleRateHz,
      stream.name,
      stream.modality,
      nSurrogates,
      "pseudo_pair"
    )
  );
}
