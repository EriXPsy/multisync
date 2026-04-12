/**
 * Cascade Analysis Module
 * 
 * Implements three complementary methods to detect which modality "leads"
 * the emergence of multimodal synchrony:
 * 
 * 1. **Onset Detection** — When does each modality's synchrony first cross
 *    a threshold? The earliest onset is the "leading" modality.
 *    (Gordon & Tomashin 2025; Burns et al. 2025)
 * 
 * 2. **Cross-Correlation Lead-Lag** — Pairwise cross-correlation of modality
 *    synchrony timeseries. A positive optimal lag means X leads Y.
 *    (Boker et al. 2002; Behrens et al. 2020)
 * 
 * 3. **Granger-style Predictive Causality** — Does adding past values of
 *    modality X improve prediction of modality Y beyond Y's own history?
 *    Implemented as a bivariate VAR(1) comparison via residual variance ratio.
 *    Includes Bonferroni correction and effect size reporting.
 *    (Granger 1969; adapted for interpersonal synchrony per Quan et al. 2025)
 * 
 * 4. **Onset Sensitivity Analysis** — Sweeps thresholds from 0.25σ to 1.5σ
 *    and reports how stable the cascade ordering is across thresholds.
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
  pValue: number;            // approximate p-value from F-distribution
  pValueCorrected: number;   // Bonferroni-corrected p-value
  varianceRatio: number;
  effectSize: number;        // partial eta-squared
  direction: "causes" | "no_effect";
  underpowered: boolean;     // true if n < 30 epochs
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
  cascadeStability: number;  // 0-1: fraction of thresholds agreeing on leader
  summary: string;
  thresholdSigma: number;
  warnings: string[];
}

/**
 * Approximate upper-tail p-value for F-distribution using a simple
 * Gaussian approximation (adequate for screening, not publication).
 */
function approxFPValue(f: number, df1: number, df2: number): number {
  if (df1 <= 0 || df2 <= 0 || f <= 0) return 1;
  // Approximation via Wilson-Hilferty transformation
  const a = df1 * f / (df1 * f + df2);
  const z = ((1 - 2 / (9 * df2)) * Math.pow(a / (1 - a) * df2 / df1, 1/3) - (1 - 2 / (9 * df1))) /
            Math.sqrt(2 / (9 * df1) + 2 / (9 * df2) * Math.pow(a / (1 - a) * df2 / df1, 2/3));
  // Standard normal CDF approximation
  const p = 0.5 * (1 - Math.tanh(z * 0.7978845608));
  return Math.max(0, Math.min(1, p));
}

/**
 * Detect onset with sustained criterion (≥2 consecutive epochs above threshold)
 */
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

/**
 * Pairwise cross-correlation with optimal lag detection.
 */
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
        let sum = 0;
        let count = 0;
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

/**
 * Granger causality with:
 * - Approximate p-values from F-distribution
 * - Bonferroni correction for multiple comparisons
 * - Partial eta-squared effect size
 * - Underpowered warnings when n < 30
 */
function grangerCausality(
  modalityTimeseries: Record<string, number[]>
): GrangerResult[] {
  const mods = Object.keys(modalityTimeseries);
  const results: GrangerResult[] = [];
  const numTests = mods.length * (mods.length - 1); // for Bonferroni

  for (const cause of mods) {
    for (const effect of mods) {
      if (cause === effect) continue;

      const x = modalityTimeseries[cause];
      const y = modalityTimeseries[effect];
      const n = Math.min(x.length, y.length);
      const underpowered = n < 30;

      if (n < 6) {
        results.push({
          cause, effect, fStatistic: 0, pValue: 1, pValueCorrected: 1,
          varianceRatio: 1, effectSize: 0, direction: "no_effect", underpowered: true,
        });
        continue;
      }

      // Restricted model: Y(t) = a * Y(t-1)
      let sumYY = 0, sumY2 = 0;
      for (let t = 1; t < n; t++) {
        sumYY += y[t] * y[t - 1];
        sumY2 += y[t - 1] ** 2;
      }
      const a_r = sumY2 !== 0 ? sumYY / sumY2 : 0;

      let rssRestricted = 0;
      for (let t = 1; t < n; t++) {
        rssRestricted += (y[t] - a_r * y[t - 1]) ** 2;
      }

      // Unrestricted model: Y(t) = a * Y(t-1) + b * X(t-1)
      let s11 = 0, s12 = 0, s22 = 0, r1 = 0, r2 = 0;
      for (let t = 1; t < n; t++) {
        const y1 = y[t - 1], x1 = x[t - 1], yt = y[t];
        s11 += y1 * y1; s12 += y1 * x1; s22 += x1 * x1;
        r1 += yt * y1; r2 += yt * x1;
      }

      const det = s11 * s22 - s12 * s12;
      let a_u = 0, b_u = 0;
      if (Math.abs(det) > 1e-12) {
        a_u = (s22 * r1 - s12 * r2) / det;
        b_u = (s11 * r2 - s12 * r1) / det;
      }

      let rssUnrestricted = 0;
      for (let t = 1; t < n; t++) {
        rssUnrestricted += (y[t] - a_u * y[t - 1] - b_u * x[t - 1]) ** 2;
      }

      const varRatio = rssUnrestricted > 0 ? rssRestricted / rssUnrestricted : 1;
      const dfNum = 1;
      const dfDen = n - 3;
      const fStat = dfDen > 0 ? ((rssRestricted - rssUnrestricted) / dfNum) / (rssUnrestricted / dfDen) : 0;

      // Approximate p-value and Bonferroni correction
      const pValue = approxFPValue(fStat, dfNum, dfDen);
      const pValueCorrected = Math.min(1, pValue * numTests);

      // Partial eta-squared effect size
      const etaSq = dfDen > 0 ? (fStat * dfNum) / (fStat * dfNum + dfDen) : 0;

      // Use corrected p-value for significance
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
      });
    }
  }

  return results;
}

/**
 * Sensitivity analysis: sweep thresholds and check cascade order stability.
 */
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

  // Stability = fraction of thresholds that agree on the leading modality
  const stability = leaders.length > 0
    ? Math.max(...Object.values(
        leaders.reduce((acc, l) => { acc[l] = (acc[l] || 0) + 1; return acc; }, {} as Record<string, number>)
      )) / leaders.length
    : 0;

  return { results, stability: parseFloat(stability.toFixed(2)) };
}

/**
 * Main entry point: run full cascade analysis on epoch-aggregated synchrony data.
 */
export function runCascadeAnalysis(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  thresholdSigma: number = 0.5
): CascadeReport {
  const warnings: string[] = [];
  const mods = Object.keys(modalityTimeseries);
  const minEpochs = Math.min(...Object.values(modalityTimeseries).map((v) => v.length));

  // Sample size warning
  if (minEpochs < 30) {
    warnings.push(
      `Low epoch count (n=${minEpochs}). Granger causality and lead-lag estimates may be unreliable. Consider using a shorter common epoch to increase sample size, or interpret results with caution.`
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

  // Cascade order from onset detection
  const cascadeOrder = [...onsets]
    .filter((o) => o.onsetEpoch !== null)
    .sort((a, b) => (a.onsetEpoch ?? Infinity) - (b.onsetEpoch ?? Infinity))
    .map((o) => o.modality);
  const neverOnset = onsets.filter((o) => o.onsetEpoch === null).map((o) => o.modality);

  // Stability warning
  if (sensitivity.stability < 0.5 && cascadeOrder.length > 1) {
    warnings.push(
      `Cascade order is unstable across thresholds (stability=${(sensitivity.stability * 100).toFixed(0)}%). The leading modality changes depending on the threshold chosen — interpret onset ordering with caution.`
    );
  }

  // Multiple comparison warning
  const numSignificant = grangerResults.filter((g) => g.direction === "causes").length;
  const numTests = mods.length * (mods.length - 1);
  if (numSignificant > 0) {
    warnings.push(
      `Granger tests: ${numSignificant}/${numTests} significant after Bonferroni correction (α=0.05, corrected α=${(0.05 / numTests).toFixed(4)}).`
    );
  }

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
      `Strongest directional influence: ${strongestGranger.cause} → ${strongestGranger.effect} (F=${strongestGranger.fStatistic.toFixed(2)}, p_corrected=${strongestGranger.pValueCorrected.toFixed(3)}, η²=${strongestGranger.effectSize.toFixed(3)}).`
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
