/**
 * Cascade Analysis Module
 * 
 * Methods:
 * 1. **Onset Detection** — sustained threshold crossing
 * 2. **Cross-Correlation Lead-Lag** — pairwise optimal lag
 * 3. **Granger-style Predictive Causality** — VAR with BIC lag selection,
 *    ADF stationarity test, Bonferroni correction, effect size
 * 4. **Onset Sensitivity Analysis** — threshold sweep
 */

export interface OnsetResult {
  modality: string;
  onsetEpoch: number | null;
  onsetTimeSec: number | null;
  peakEpoch: number;
  peakValue: number;
  meanValue: number;
}

export interface LeadLagResult {
  from: string;
  to: string;
  optimalLagEpochs: number;
  optimalLagSec: number;
  peakCorrelation: number;
}

export interface GrangerResult {
  cause: string;
  effect: string;
  fStatistic: number;
  pValue: number;
  pValueCorrected: number;
  varianceRatio: number;
  effectSize: number;         // partial eta-squared
  direction: "causes" | "no_effect";
  underpowered: boolean;
  selectedLag: number;        // BIC-selected lag order
  stationaryX: boolean;       // ADF test result for cause
  stationaryY: boolean;       // ADF test result for effect
  differenced: boolean;       // whether data was differenced
}

export interface SensitivityResult {
  threshold: number;
  cascadeOrder: string[];
}

export interface CascadeReport {
  onsets: OnsetResult[];
  cascadeOrder: string[];
  leadLagMatrix: LeadLagResult[];
  grangerResults: GrangerResult[];
  sensitivityAnalysis: SensitivityResult[];
  cascadeStability: number;
  summary: string;
  thresholdSigma: number;
  warnings: string[];
}

// ─── Statistical Utilities ───────────────────────────────────────

/**
 * Approximate upper-tail p-value for F-distribution (Wilson-Hilferty).
 * NOTE: This is an approximation adequate for exploratory screening,
 * NOT for confirmatory publication-quality inference.
 */
function approxFPValue(f: number, df1: number, df2: number): number {
  if (df1 <= 0 || df2 <= 0 || f <= 0) return 1;
  const a = df1 * f / (df1 * f + df2);
  const z = ((1 - 2 / (9 * df2)) * Math.pow(a / (1 - a) * df2 / df1, 1/3) - (1 - 2 / (9 * df1))) /
            Math.sqrt(2 / (9 * df1) + 2 / (9 * df2) * Math.pow(a / (1 - a) * df2 / df1, 2/3));
  const p = 0.5 * (1 - Math.tanh(z * 0.7978845608));
  return Math.max(0, Math.min(1, p));
}

/**
 * Augmented Dickey-Fuller test (simplified).
 * Tests H0: unit root (non-stationary) vs H1: stationary.
 * Returns true if series appears stationary (reject H0 at ~5% level).
 * 
 * Uses ADF regression: Δy_t = α + β*y_{t-1} + Σ γ_i*Δy_{t-i} + ε
 * Critical value ≈ -2.86 for n≈30 at 5% (MacKinnon, 1996).
 */
function adfTest(series: number[], maxAdfLag: number = 2): boolean {
  const n = series.length;
  if (n < 10) return false; // too short to test

  // First differences
  const dy: number[] = [];
  for (let i = 1; i < n; i++) dy.push(series[i] - series[i - 1]);

  const lag = Math.min(maxAdfLag, Math.floor(dy.length / 4));
  const start = lag + 1;
  const T = dy.length - start;
  if (T < 6) return false;

  // Simple ADF: regress Δy_t on y_{t-1} (and intercept)
  // t-statistic for β coefficient
  let sumY = 0, sumY2 = 0, sumDyY = 0, sumDy = 0;
  for (let t = start; t < dy.length; t++) {
    const yLag = series[t]; // y_{t-1} since dy[t] = series[t+1] - series[t]
    sumY += yLag;
    sumY2 += yLag * yLag;
    sumDyY += dy[t] * yLag;
    sumDy += dy[t];
  }

  const meanY = sumY / T;
  const meanDy = sumDy / T;
  const ssY = sumY2 - T * meanY * meanY;
  if (ssY < 1e-12) return true; // constant series is stationary

  const beta = (sumDyY - T * meanDy * meanY) / ssY;

  // Residual variance
  let rss = 0;
  for (let t = start; t < dy.length; t++) {
    const yLag = series[t];
    const predicted = meanDy + beta * (yLag - meanY);
    rss += (dy[t] - predicted) ** 2;
  }
  const se = Math.sqrt(rss / (T - 2));
  const seBeta = se / Math.sqrt(ssY);
  
  if (seBeta < 1e-12) return true;
  const tStat = beta / seBeta;

  // Critical value at 5% for ADF with intercept, n≈30
  const criticalValue = -2.86;
  return tStat < criticalValue;
}

/**
 * First-difference a series to achieve stationarity.
 */
function difference(series: number[]): number[] {
  const result: number[] = [];
  for (let i = 1; i < series.length; i++) {
    result.push(series[i] - series[i - 1]);
  }
  return result;
}

/**
 * Compute BIC for a VAR model RSS.
 * BIC = n * ln(RSS/n) + k * ln(n)
 */
function computeBIC(rss: number, n: number, k: number): number {
  if (n <= 0 || rss <= 0) return Infinity;
  return n * Math.log(rss / n) + k * Math.log(n);
}

// ─── Onset Detection ─────────────────────────────────────────────

function detectOnsets(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  thresholdSigma: number = 0.5
): OnsetResult[] {
  const results: OnsetResult[] = [];

  for (const [mod, values] of Object.entries(modalityTimeseries)) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const peakValue = Math.max(...values);
    const peakEpoch = values.indexOf(peakValue);

    let onsetEpoch: number | null = null;
    for (let i = 0; i < values.length - 1; i++) {
      if (values[i] >= thresholdSigma && values[i + 1] >= thresholdSigma) {
        onsetEpoch = i;
        break;
      }
    }

    results.push({
      modality: mod,
      onsetEpoch,
      onsetTimeSec: onsetEpoch !== null ? (onsetEpoch * epochMs) / 1000 : null,
      peakEpoch,
      peakValue,
      meanValue: parseFloat(mean.toFixed(4)),
    });
  }

  return results;
}

// ─── Cross-Correlation Lead-Lag ──────────────────────────────────

function computeLeadLag(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  maxLagEpochs: number = 5
): LeadLagResult[] {
  const mods = Object.keys(modalityTimeseries);
  const results: LeadLagResult[] = [];

  for (let i = 0; i < mods.length; i++) {
    for (let j = i + 1; j < mods.length; j++) {
      const x = modalityTimeseries[mods[i]];
      const y = modalityTimeseries[mods[j]];
      const n = Math.min(x.length, y.length);
      if (n < 4) continue;

      const mx = x.slice(0, n).reduce((a, b) => a + b, 0) / n;
      const my = y.slice(0, n).reduce((a, b) => a + b, 0) / n;
      const sx = Math.sqrt(x.slice(0, n).reduce((a, b) => a + (b - mx) ** 2, 0) / n);
      const sy = Math.sqrt(y.slice(0, n).reduce((a, b) => a + (b - my) ** 2, 0) / n);

      if (sx === 0 || sy === 0) continue;

      let bestLag = 0;
      let bestCorr = -Infinity;

      for (let lag = -maxLagEpochs; lag <= maxLagEpochs; lag++) {
        let sum = 0, count = 0;
        for (let t = 0; t < n; t++) {
          const t2 = t + lag;
          if (t2 >= 0 && t2 < n) {
            sum += ((x[t] - mx) / sx) * ((y[t2] - my) / sy);
            count++;
          }
        }
        const corr = count > 0 ? sum / count : 0;
        if (corr > bestCorr) {
          bestCorr = corr;
          bestLag = lag;
        }
      }

      const from = bestLag >= 0 ? mods[i] : mods[j];
      const to = bestLag >= 0 ? mods[j] : mods[i];

      results.push({
        from,
        to,
        optimalLagEpochs: Math.abs(bestLag),
        optimalLagSec: parseFloat(((Math.abs(bestLag) * epochMs) / 1000).toFixed(1)),
        peakCorrelation: parseFloat(bestCorr.toFixed(4)),
      });
    }
  }

  return results;
}

// ─── Granger Causality with BIC + ADF ────────────────────────────

/**
 * Fit a restricted AR(p) model: Y(t) = Σ a_i * Y(t-i) + ε
 * Returns RSS.
 */
function fitAR(y: number[], p: number): number {
  const n = y.length;
  if (n <= p + 1) return Infinity;

  // Simple OLS for AR(p) — normal equations via iterative sums
  // For simplicity with variable lag, use direct residual computation
  // with coefficient estimation via pseudo-inverse approach
  const T = n - p;
  
  // Build X matrix columns and solve via least squares
  // For small p (≤10), direct computation is fine
  const predictions = new Float64Array(T);
  
  // Estimate coefficients using Yule-Walker-like approach
  // Compute autocovariances
  const mean = y.reduce((a, b) => a + b, 0) / n;
  const centered = y.map(v => v - mean);
  
  const gamma: number[] = [];
  for (let k = 0; k <= p; k++) {
    let sum = 0;
    for (let t = k; t < n; t++) {
      sum += centered[t] * centered[t - k];
    }
    gamma.push(sum / n);
  }
  
  if (gamma[0] < 1e-12) return 0; // no variance
  
  // Levinson-Durbin recursion for AR coefficients
  const coeffs = levinsonDurbin(gamma, p);
  
  // Compute RSS
  let rss = 0;
  for (let t = p; t < n; t++) {
    let pred = mean;
    for (let i = 0; i < p; i++) {
      pred += coeffs[i] * (y[t - 1 - i] - mean);
    }
    rss += (y[t] - pred) ** 2;
  }
  
  return rss;
}

/**
 * Fit unrestricted VAR: Y(t) = Σ a_i*Y(t-i) + Σ b_i*X(t-i) + ε
 * Returns RSS.
 */
function fitVAR(y: number[], x: number[], p: number): number {
  const n = Math.min(y.length, x.length);
  if (n <= 2 * p + 1) return Infinity;

  const T = n - p;
  const mean_y = y.slice(0, n).reduce((a, b) => a + b, 0) / n;
  const mean_x = x.slice(0, n).reduce((a, b) => a + b, 0) / n;

  // For small p, use direct OLS via normal equations
  // Design matrix: [y_{t-1}...y_{t-p}, x_{t-1}...x_{t-p}]
  const k = 2 * p; // number of regressors
  
  // Gram matrix and cross-product vector (XtX and Xty)
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty: number[] = new Array(k).fill(0);

  for (let t = p; t < n; t++) {
    const regressors: number[] = [];
    for (let i = 1; i <= p; i++) regressors.push(y[t - i] - mean_y);
    for (let i = 1; i <= p; i++) regressors.push(x[t - i] - mean_x);

    const target = y[t] - mean_y;
    for (let a = 0; a < k; a++) {
      Xty[a] += regressors[a] * target;
      for (let b = 0; b < k; b++) {
        XtX[a][b] += regressors[a] * regressors[b];
      }
    }
  }

  // Solve via Gaussian elimination
  const coeffs = solveLinearSystem(XtX, Xty);
  if (!coeffs) return fitAR(y, p); // fallback if singular

  // Compute RSS
  let rss = 0;
  for (let t = p; t < n; t++) {
    const regressors: number[] = [];
    for (let i = 1; i <= p; i++) regressors.push(y[t - i] - mean_y);
    for (let i = 1; i <= p; i++) regressors.push(x[t - i] - mean_x);

    let pred = mean_y;
    for (let i = 0; i < k; i++) pred += coeffs[i] * regressors[i];
    rss += (y[t] - pred) ** 2;
  }

  return rss;
}

/** Levinson-Durbin recursion for AR coefficients */
function levinsonDurbin(gamma: number[], p: number): number[] {
  if (p === 0 || gamma[0] < 1e-12) return [];
  
  let coeffs = [gamma[1] / gamma[0]];
  let error = gamma[0] * (1 - coeffs[0] * coeffs[0]);

  for (let m = 1; m < p; m++) {
    let sum = gamma[m + 1];
    for (let j = 0; j < m; j++) {
      sum += coeffs[j] * gamma[m - j];
    }
    if (Math.abs(error) < 1e-12) break;
    const k = -sum / error;
    
    const newCoeffs = new Array(m + 1);
    for (let j = 0; j < m; j++) {
      newCoeffs[j] = coeffs[j] + k * coeffs[m - 1 - j];
    }
    newCoeffs[m] = k;
    coeffs = newCoeffs;
    error *= (1 - k * k);
  }

  return coeffs;
}

/** Gaussian elimination for small linear systems */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-12) return null;

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = aug[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= aug[i][j] * x[j];
    x[i] /= aug[i][i];
  }
  return x;
}

function grangerCausality(
  modalityTimeseries: Record<string, number[]>
): GrangerResult[] {
  const mods = Object.keys(modalityTimeseries);
  const results: GrangerResult[] = [];
  const numTests = mods.length * (mods.length - 1);
  const maxLagSearch = 10;

  for (const cause of mods) {
    for (const effect of mods) {
      if (cause === effect) continue;

      let x = [...modalityTimeseries[cause]];
      let y = [...modalityTimeseries[effect]];
      const n = Math.min(x.length, y.length);
      x = x.slice(0, n);
      y = y.slice(0, n);

      const underpowered = n < 30;

      if (n < 8) {
        results.push({
          cause, effect, fStatistic: 0, pValue: 1, pValueCorrected: 1,
          varianceRatio: 1, effectSize: 0, direction: "no_effect",
          underpowered: true, selectedLag: 1, stationaryX: false,
          stationaryY: false, differenced: false,
        });
        continue;
      }

      // ADF stationarity test
      const stationaryX = adfTest(x);
      const stationaryY = adfTest(y);
      let differenced = false;

      // If non-stationary, difference the series
      if (!stationaryX || !stationaryY) {
        x = difference(x);
        y = difference(y);
        differenced = true;
        if (x.length < 8) {
          results.push({
            cause, effect, fStatistic: 0, pValue: 1, pValueCorrected: 1,
            varianceRatio: 1, effectSize: 0, direction: "no_effect",
            underpowered: true, selectedLag: 1, stationaryX, stationaryY,
            differenced: true,
          });
          continue;
        }
      }

      // BIC-based lag selection (1 to maxLagSearch)
      const effectiveN = x.length;
      const maxLag = Math.min(maxLagSearch, Math.floor(effectiveN / 4));
      let bestLag = 1;
      let bestBIC = Infinity;

      for (let p = 1; p <= maxLag; p++) {
        const rssR = fitAR(y, p);
        if (!isFinite(rssR)) continue;
        const T = effectiveN - p;
        const bic = computeBIC(rssR, T, p + 1); // p coeffs + intercept
        if (bic < bestBIC) {
          bestBIC = bic;
          bestLag = p;
        }
      }

      // Fit restricted (AR) and unrestricted (VAR) at selected lag
      const rssRestricted = fitAR(y, bestLag);
      const rssUnrestricted = fitVAR(y, x, bestLag);

      const T = effectiveN - bestLag;
      const dfNum = bestLag;
      const dfDen = T - 2 * bestLag - 1;

      let fStat = 0;
      if (dfDen > 0 && rssUnrestricted > 0 && rssRestricted > rssUnrestricted) {
        fStat = ((rssRestricted - rssUnrestricted) / dfNum) / (rssUnrestricted / dfDen);
      }

      const varRatio = rssUnrestricted > 0 ? rssRestricted / rssUnrestricted : 1;
      const pValue = approxFPValue(fStat, dfNum, dfDen);
      const pValueCorrected = Math.min(1, pValue * numTests);
      const etaSq = dfDen > 0 ? (fStat * dfNum) / (fStat * dfNum + dfDen) : 0;

      const significant = pValueCorrected < 0.05 && varRatio > 1.05;

      results.push({
        cause,
        effect,
        fStatistic: parseFloat(fStat.toFixed(3)),
        pValue: parseFloat(pValue.toFixed(4)),
        pValueCorrected: parseFloat(pValueCorrected.toFixed(4)),
        varianceRatio: parseFloat(varRatio.toFixed(4)),
        effectSize: parseFloat(etaSq.toFixed(4)),
        direction: significant ? "causes" : "no_effect",
        underpowered,
        selectedLag: bestLag,
        stationaryX,
        stationaryY,
        differenced,
      });
    }
  }

  return results;
}

// ─── Sensitivity Analysis ────────────────────────────────────────

function onsetSensitivityAnalysis(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number
): { results: SensitivityResult[]; stability: number } {
  const thresholds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5];
  const results: SensitivityResult[] = [];
  const leaders: string[] = [];

  for (const thresh of thresholds) {
    const onsets = detectOnsets(modalityTimeseries, epochMs, thresh);
    const order = [...onsets]
      .filter((o) => o.onsetEpoch !== null)
      .sort((a, b) => (a.onsetEpoch ?? Infinity) - (b.onsetEpoch ?? Infinity))
      .map((o) => o.modality);

    const neverOnset = onsets.filter((o) => o.onsetEpoch === null).map((o) => o.modality);
    results.push({ threshold: thresh, cascadeOrder: [...order, ...neverOnset] });

    if (order.length > 0) leaders.push(order[0]);
  }

  const stability = leaders.length > 0
    ? Math.max(...Object.values(
        leaders.reduce((acc, l) => { acc[l] = (acc[l] || 0) + 1; return acc; }, {} as Record<string, number>)
      )) / leaders.length
    : 0;

  return { results, stability: parseFloat(stability.toFixed(2)) };
}

// ─── Main Entry Point ────────────────────────────────────────────

export function runCascadeAnalysis(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  thresholdSigma: number = 0.5
): CascadeReport {
  const warnings: string[] = [];
  const mods = Object.keys(modalityTimeseries);
  const minEpochs = Math.min(...Object.values(modalityTimeseries).map((v) => v.length));

  if (minEpochs < 30) {
    warnings.push(
      `Low epoch count (n=${minEpochs}). Granger causality and lead-lag estimates may be unreliable. Consider shorter epochs for more statistical power.`
    );
  }
  if (minEpochs < 10) {
    warnings.push(
      `Very low epoch count (n=${minEpochs}). Statistical tests are severely underpowered. Results should be considered exploratory only.`
    );
  }

  const onsets = detectOnsets(modalityTimeseries, epochMs, thresholdSigma);
  const leadLagMatrix = computeLeadLag(modalityTimeseries, epochMs);
  const grangerResults = grangerCausality(modalityTimeseries);
  const sensitivity = onsetSensitivityAnalysis(modalityTimeseries, epochMs);

  const cascadeOrder = [...onsets]
    .filter((o) => o.onsetEpoch !== null)
    .sort((a, b) => (a.onsetEpoch ?? Infinity) - (b.onsetEpoch ?? Infinity))
    .map((o) => o.modality);
  const neverOnset = onsets.filter((o) => o.onsetEpoch === null).map((o) => o.modality);

  if (sensitivity.stability < 0.5 && cascadeOrder.length > 1) {
    warnings.push(
      `Cascade order is unstable across thresholds (stability=${(sensitivity.stability * 100).toFixed(0)}%). Interpret onset ordering with caution.`
    );
  }

  // Stationarity warnings
  const nonStationary = grangerResults.filter(g => g.differenced);
  if (nonStationary.length > 0) {
    warnings.push(
      `${nonStationary.length}/${grangerResults.length} Granger pairs required differencing due to non-stationarity (ADF test). Original trends were removed before testing.`
    );
  }

  const numSignificant = grangerResults.filter((g) => g.direction === "causes").length;
  const numGrangerTests = mods.length * (mods.length - 1);
  if (numSignificant > 0) {
    warnings.push(
      `Granger tests: ${numSignificant}/${numGrangerTests} significant after Bonferroni correction (α=0.05).`
    );
  }

  // P-value approximation warning — always shown
  warnings.push(
    `⚠ P-values use Wilson-Hilferty F-approximation — suitable for exploratory screening only. For publication, validate with exact F-tables or permutation tests.`
  );

  // Build summary
  const summaryParts: string[] = [];

  if (cascadeOrder.length > 0) {
    const leader = cascadeOrder[0];
    const leaderOnset = onsets.find((o) => o.modality === leader);
    summaryParts.push(
      `Cascade order: ${cascadeOrder.map((m) => m.charAt(0).toUpperCase() + m.slice(1)).join(" → ")}.`
    );
    summaryParts.push(
      `${leader.charAt(0).toUpperCase() + leader.slice(1)} synchrony emerges first (onset at ${leaderOnset?.onsetTimeSec?.toFixed(1)}s, threshold ${thresholdSigma}σ).`
    );
    summaryParts.push(
      `Order stability: ${(sensitivity.stability * 100).toFixed(0)}% across thresholds (0.25σ–1.5σ).`
    );
  }

  if (neverOnset.length > 0) {
    summaryParts.push(`${neverOnset.join(", ")} did not reach the ${thresholdSigma}σ threshold.`);
  }

  const strongestGranger = grangerResults
    .filter((g) => g.direction === "causes")
    .sort((a, b) => b.fStatistic - a.fStatistic)[0];

  if (strongestGranger) {
    summaryParts.push(
      `Strongest directional influence: ${strongestGranger.cause} → ${strongestGranger.effect} (F=${strongestGranger.fStatistic.toFixed(2)}, p_corr=${strongestGranger.pValueCorrected.toFixed(3)}, η²=${strongestGranger.effectSize.toFixed(3)}, lag=${strongestGranger.selectedLag}).`
    );
  }

  const strongestLag = leadLagMatrix
    .filter((l) => l.optimalLagEpochs > 0)
    .sort((a, b) => b.peakCorrelation - a.peakCorrelation)[0];

  if (strongestLag) {
    summaryParts.push(
      `Strongest temporal lead: ${strongestLag.from} leads ${strongestLag.to} by ${strongestLag.optimalLagSec}s (r=${strongestLag.peakCorrelation.toFixed(3)}).`
    );
  }

  return {
    onsets,
    cascadeOrder: [...cascadeOrder, ...neverOnset],
    leadLagMatrix,
    grangerResults,
    sensitivityAnalysis: sensitivity.results,
    cascadeStability: sensitivity.stability,
    summary: summaryParts.join(" "),
    thresholdSigma,
    warnings,
  };
}
