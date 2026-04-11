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
 *    (Granger 1969; adapted for interpersonal synchrony per Quan et al. 2025)
 */

export interface OnsetResult {
  modality: string;
  onsetEpoch: number | null; // null = never crossed threshold
  onsetTimeSec: number | null;
  peakEpoch: number;
  peakValue: number;
  meanValue: number;
}

export interface LeadLagResult {
  from: string;   // "leading" modality
  to: string;     // "following" modality
  optimalLagEpochs: number; // positive = from leads to
  optimalLagSec: number;
  peakCorrelation: number;
}

export interface GrangerResult {
  cause: string;
  effect: string;
  fStatistic: number;
  varianceRatio: number; // >1 means cause improves prediction
  direction: "causes" | "no_effect";
}

export interface CascadeReport {
  onsets: OnsetResult[];
  cascadeOrder: string[];         // modalities sorted by onset time
  leadLagMatrix: LeadLagResult[];
  grangerResults: GrangerResult[];
  summary: string;                // human-readable interpretation
  thresholdSigma: number;
}

/**
 * Detect when each modality's z-scored synchrony first exceeds thresholdσ
 * Uses a sustained criterion: must stay above for ≥2 consecutive epochs
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

    // Sustained crossing: 2+ consecutive epochs above threshold
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
 * Pairwise cross-correlation between modality synchrony timeseries.
 * Returns optimal lag and correlation for each directed pair.
 * Positive lag means "from" leads "to".
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

      // Positive bestLag means x leads y
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
 * Simplified Granger causality test.
 * Compares two AR(1) models:
 *   Restricted:   Y(t) = a * Y(t-1) + ε
 *   Unrestricted: Y(t) = a * Y(t-1) + b * X(t-1) + ε
 * If adding X reduces residual variance, X "Granger-causes" Y.
 * Returns variance ratio (restricted / unrestricted). >1 = causal.
 */
function grangerCausality(
  modalityTimeseries: Record<string, number[]>
): GrangerResult[] {
  const mods = Object.keys(modalityTimeseries);
  const results: GrangerResult[] = [];

  for (const cause of mods) {
    for (const effect of mods) {
      if (cause === effect) continue;

      const x = modalityTimeseries[cause];
      const y = modalityTimeseries[effect];
      const n = Math.min(x.length, y.length);
      if (n < 6) continue;

      // Restricted model: Y(t) = a * Y(t-1)
      // Least squares: a = Σ(Y(t)*Y(t-1)) / Σ(Y(t-1)²)
      let sumYY = 0, sumY2 = 0;
      for (let t = 1; t < n; t++) {
        sumYY += y[t] * y[t - 1];
        sumY2 += y[t - 1] ** 2;
      }
      const a_r = sumY2 !== 0 ? sumYY / sumY2 : 0;

      let rssRestricted = 0;
      for (let t = 1; t < n; t++) {
        const pred = a_r * y[t - 1];
        rssRestricted += (y[t] - pred) ** 2;
      }

      // Unrestricted model: Y(t) = a * Y(t-1) + b * X(t-1)
      // 2-variable OLS via normal equations
      let s11 = 0, s12 = 0, s22 = 0, r1 = 0, r2 = 0;
      for (let t = 1; t < n; t++) {
        const y1 = y[t - 1];
        const x1 = x[t - 1];
        const yt = y[t];
        s11 += y1 * y1;
        s12 += y1 * x1;
        s22 += x1 * x1;
        r1 += yt * y1;
        r2 += yt * x1;
      }

      const det = s11 * s22 - s12 * s12;
      let a_u = 0, b_u = 0;
      if (Math.abs(det) > 1e-12) {
        a_u = (s22 * r1 - s12 * r2) / det;
        b_u = (s11 * r2 - s12 * r1) / det;
      }

      let rssUnrestricted = 0;
      for (let t = 1; t < n; t++) {
        const pred = a_u * y[t - 1] + b_u * x[t - 1];
        rssUnrestricted += (y[t] - pred) ** 2;
      }

      const varRatio = rssUnrestricted > 0 ? rssRestricted / rssUnrestricted : 1;
      const dfNum = 1;
      const dfDen = n - 3; // unrestricted has 2 params + intercept
      const fStat = dfDen > 0 ? ((rssRestricted - rssUnrestricted) / dfNum) / (rssUnrestricted / dfDen) : 0;

      results.push({
        cause,
        effect,
        fStatistic: parseFloat(fStat.toFixed(3)),
        varianceRatio: parseFloat(varRatio.toFixed(4)),
        direction: varRatio > 1.05 && fStat > 2.0 ? "causes" : "no_effect",
      });
    }
  }

  return results;
}

/**
 * Main entry point: run full cascade analysis on epoch-aggregated synchrony data.
 */
export function runCascadeAnalysis(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  thresholdSigma: number = 0.5
): CascadeReport {
  const onsets = detectOnsets(modalityTimeseries, epochMs, thresholdSigma);
  const leadLagMatrix = computeLeadLag(modalityTimeseries, epochMs);
  const grangerResults = grangerCausality(modalityTimeseries);

  // Sort by onset time to get cascade order
  const cascadeOrder = [...onsets]
    .filter((o) => o.onsetEpoch !== null)
    .sort((a, b) => (a.onsetEpoch ?? Infinity) - (b.onsetEpoch ?? Infinity))
    .map((o) => o.modality);

  // Never-onset modalities appended at end
  const neverOnset = onsets.filter((o) => o.onsetEpoch === null).map((o) => o.modality);

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
  }

  if (neverOnset.length > 0) {
    summaryParts.push(
      `${neverOnset.join(", ")} did not reach the ${thresholdSigma}σ threshold.`
    );
  }

  // Strongest Granger causal link
  const strongestGranger = grangerResults
    .filter((g) => g.direction === "causes")
    .sort((a, b) => b.fStatistic - a.fStatistic)[0];

  if (strongestGranger) {
    summaryParts.push(
      `Strongest directional influence: ${strongestGranger.cause} → ${strongestGranger.effect} (F = ${strongestGranger.fStatistic.toFixed(2)}, variance ratio = ${strongestGranger.varianceRatio.toFixed(2)}).`
    );
  }

  // Strongest lead-lag
  const strongestLag = leadLagMatrix
    .filter((l) => l.optimalLagEpochs > 0)
    .sort((a, b) => b.peakCorrelation - a.peakCorrelation)[0];

  if (strongestLag) {
    summaryParts.push(
      `Strongest temporal lead: ${strongestLag.from} leads ${strongestLag.to} by ${strongestLag.optimalLagSec}s (r = ${strongestLag.peakCorrelation.toFixed(3)}).`
    );
  }

  return {
    onsets,
    cascadeOrder: [...cascadeOrder, ...neverOnset],
    leadLagMatrix,
    grangerResults,
    summary: summaryParts.join(" "),
    thresholdSigma,
  };
}
