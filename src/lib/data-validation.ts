/**
 * Data Validation Module
 * 
 * Validates imported dyadic timeseries data for common issues
 * that could compromise analysis validity.
 */

export interface ValidationWarning {
  severity: "error" | "warning" | "info";
  message: string;
  field?: string;
}

export interface ValidationReport {
  valid: boolean;
  warnings: ValidationWarning[];
  stats: {
    totalRows: number;
    missingPercent: number;
    gapCount: number;
    timestampMonotonic: boolean;
    duplicateTimestamps: number;
    p1Variance: number;
    p2Variance: number;
    estimatedSampleRate: number | null;
  };
}

/**
 * Validate a dyadic timeseries dataset.
 */
export function validateDyadicData(
  data: Array<{ t: number; p1: number; p2: number }>,
  expectedSampleRate?: number
): ValidationReport {
  const warnings: ValidationWarning[] = [];
  const n = data.length;

  if (n === 0) {
    return {
      valid: false,
      warnings: [{ severity: "error", message: "Dataset is empty — no rows found." }],
      stats: { totalRows: 0, missingPercent: 100, gapCount: 0, timestampMonotonic: false, duplicateTimestamps: 0, p1Variance: 0, p2Variance: 0, estimatedSampleRate: null },
    };
  }

  if (n < 10) {
    warnings.push({ severity: "error", message: `Only ${n} data points — too few for meaningful synchrony analysis (minimum ~30 recommended).` });
  }

  // Check for NaN/Infinity values
  let nanCount = 0;
  let infCount = 0;
  for (const d of data) {
    if (isNaN(d.p1) || isNaN(d.p2) || isNaN(d.t)) nanCount++;
    if (!isFinite(d.p1) || !isFinite(d.p2)) infCount++;
  }
  const missingPercent = (nanCount / n) * 100;
  if (nanCount > 0) {
    warnings.push({
      severity: missingPercent > 20 ? "error" : "warning",
      message: `${nanCount} rows (${missingPercent.toFixed(1)}%) contain NaN/missing values. ${missingPercent > 20 ? "This exceeds 20% — consider cleaning data first." : "These will be treated as zeros."}`,
    });
  }
  if (infCount > 0) {
    warnings.push({ severity: "error", message: `${infCount} rows contain Infinity values — data may be corrupted.` });
  }

  // Check timestamp monotonicity
  let monotonic = true;
  let duplicateTs = 0;
  let gapCount = 0;
  const intervals: number[] = [];

  for (let i = 1; i < n; i++) {
    const dt = data[i].t - data[i - 1].t;
    if (dt < 0) monotonic = false;
    if (dt === 0) duplicateTs++;
    if (dt > 0) intervals.push(dt);
  }

  if (!monotonic) {
    warnings.push({ severity: "error", message: "Timestamps are not monotonically increasing — data may be unsorted or corrupted.", field: "timestamp" });
  }
  if (duplicateTs > 0) {
    warnings.push({ severity: "warning", message: `${duplicateTs} duplicate timestamps detected. These may cause artifacts in WCC computation.`, field: "timestamp" });
  }

  // Estimate sample rate and check for gaps
  let estimatedSampleRate: number | null = null;
  if (intervals.length > 0) {
    intervals.sort((a, b) => a - b);
    const medianInterval = intervals[Math.floor(intervals.length / 2)];
    estimatedSampleRate = medianInterval > 0 ? 1000 / medianInterval : null;

    // Gaps: intervals > 3x median
    for (const dt of intervals) {
      if (dt > medianInterval * 3 && medianInterval > 0) gapCount++;
    }

    if (gapCount > 0) {
      warnings.push({
        severity: "warning",
        message: `${gapCount} temporal gaps detected (intervals >3x median). These may indicate recording interruptions.`,
        field: "timestamp",
      });
    }

    if (expectedSampleRate && estimatedSampleRate) {
      const ratio = estimatedSampleRate / expectedSampleRate;
      if (ratio < 0.5 || ratio > 2) {
        warnings.push({
          severity: "warning",
          message: `Estimated sample rate (${estimatedSampleRate.toFixed(1)}Hz) differs significantly from expected (${expectedSampleRate}Hz). Check time unit settings.`,
          field: "sampleRate",
        });
      }
    }
  }

  // Check signal variance (zero-variance = flat line = no information)
  const p1Mean = data.reduce((a, d) => a + d.p1, 0) / n;
  const p2Mean = data.reduce((a, d) => a + d.p2, 0) / n;
  const p1Var = data.reduce((a, d) => a + (d.p1 - p1Mean) ** 2, 0) / n;
  const p2Var = data.reduce((a, d) => a + (d.p2 - p2Mean) ** 2, 0) / n;

  if (p1Var === 0) {
    warnings.push({ severity: "error", message: "Person 1 signal has zero variance (flat line) — no meaningful synchrony can be computed.", field: "p1" });
  }
  if (p2Var === 0) {
    warnings.push({ severity: "error", message: "Person 2 signal has zero variance (flat line) — no meaningful synchrony can be computed.", field: "p2" });
  }

  // Check if signals are identical (perfect correlation = suspicious)
  if (p1Var > 0 && p2Var > 0) {
    let allSame = true;
    for (let i = 0; i < Math.min(n, 100); i++) {
      if (Math.abs(data[i].p1 - data[i].p2) > 1e-10) { allSame = false; break; }
    }
    if (allSame) {
      warnings.push({ severity: "warning", message: "Person 1 and Person 2 signals appear identical — this may indicate the same signal was mapped to both columns." });
    }
  }

  // Duration check
  const duration = n > 0 ? data[n - 1].t - data[0].t : 0;
  if (duration < 30000) {
    warnings.push({ severity: "info", message: `Recording duration is ${(duration / 1000).toFixed(1)}s — very short for synchrony analysis. Results may be unreliable.` });
  }

  const hasErrors = warnings.some(w => w.severity === "error");

  return {
    valid: !hasErrors,
    warnings,
    stats: {
      totalRows: n,
      missingPercent: parseFloat(missingPercent.toFixed(1)),
      gapCount,
      timestampMonotonic: monotonic,
      duplicateTimestamps: duplicateTs,
      p1Variance: parseFloat(p1Var.toFixed(4)),
      p2Variance: parseFloat(p2Var.toFixed(4)),
      estimatedSampleRate: estimatedSampleRate ? parseFloat(estimatedSampleRate.toFixed(1)) : null,
    },
  };
}
