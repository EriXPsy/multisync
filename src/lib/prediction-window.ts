/**
 * Prediction Window Analysis Module
 * 
 * Tests whether synchrony in one modality at time t predicts
 * multimodal synchrony at time t + Δt.
 * 
 * This is the "prelude signal" detection — finding which notes
 * predict the full chord.
 * 
 * Methods:
 * 1. Logistic Regression (simple, interpretable)
 * 2. Sliding window feature extraction
 * 3. Leave-one-epoch-out cross-validation
 * 4. Feature importance via coefficient magnitude
 * 
 * NOTE: This is a proof-of-concept implementation for browser-side use.
 * For publication-quality results, export features and run scikit-learn
 * in Python with proper cross-validation.
 */

export interface PredictionConfig {
  deltaT: number;         // prediction horizon in epochs (e.g., 2 = predict 2 epochs ahead)
  thresholdSigma: number; // threshold for "high synchrony" label
  useChangeRate: boolean; // include derivative features
  useAcceleration: boolean; // include second-derivative features
}

export interface PredictionFeatures {
  epoch: number;
  timeSec: number;
  // Per-modality features
  features: Record<string, number[]>;
  // Label: will high synchrony occur at t + deltaT?
  label: number; // 1 = yes, 0 = no
  // Actual outcome for evaluation
  actualOutcome: number;
}

export interface PredictionResult {
  model: string;
  deltaT: number;
  deltaTSec: number;
  totalSamples: number;
  positiveSamples: number;
  negativeSamples: number;
  // Performance metrics (leave-one-out)
  accuracy: number;
  sensitivity: number;  // true positive rate (recall)
  specificity: number;  // true negative rate
  precision: number;
  f1: number;
  auc: number;          // approximate AUC via ranking
  // Feature importance
  featureImportance: { feature: string; weight: number; description: string }[];
  // Confusion matrix
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  // Warnings
  warnings: string[];
  summary: string;
}

export interface MultiDyadPredictionResult {
  dyadPredictions: { dyadId: string; result: PredictionResult }[];
  aggregatePerformance: {
    meanAccuracy: number;
    meanF1: number;
    meanAUC: number;
    meanSensitivity: number;
    meanPrecision: number;
    consistentFeatures: { feature: string; meanWeight: number; frequency: number }[];
  };
  warnings: string[];
  summary: string;
}

const DEFAULT_CONFIG: PredictionConfig = {
  deltaT: 2,
  thresholdSigma: 0.5,
  useChangeRate: true,
  useAcceleration: false,
};

/**
 * Sigmoid function for logistic regression.
 */
function sigmoid(z: number): number {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

/**
 * Extract features from a sliding window for prediction.
 */
function extractWindowFeatures(
  modalityTimeseries: Record<string, number[]>,
  windowEpoch: number,
  config: PredictionConfig
): Record<string, number> {
  const features: Record<string, number> = {};
  const mods = Object.keys(modalityTimeseries);

  for (const mod of mods) {
    const vals = modalityTimeseries[mod];
    if (windowEpoch < 0 || windowEpoch >= vals.length) continue;

    const current = vals[windowEpoch];
    features[`${mod}_value`] = current;

    // Change rate (first derivative)
    if (config.useChangeRate && windowEpoch >= 1) {
      features[`${mod}_changeRate`] = vals[windowEpoch] - vals[windowEpoch - 1];
    }

    // Acceleration (second derivative)
    if (config.useAcceleration && windowEpoch >= 2) {
      features[`${mod}_acceleration`] =
        vals[windowEpoch] - 2 * vals[windowEpoch - 1] + vals[windowEpoch - 2];
    }

    // Short-term trend (moving average deviation)
    if (windowEpoch >= 3) {
      const recent = vals.slice(windowEpoch - 3, windowEpoch);
      const ma = recent.reduce((a, b) => a + b, 0) / recent.length;
      features[`${mod}_trendDev`] = current - ma;
    }

    // Cross-modality features: difference from other modalities
    for (const other of mods) {
      if (other === mod) continue;
      const otherVals = modalityTimeseries[other];
      if (windowEpoch < otherVals.length) {
        features[`${mod}_minus_${other}`] = current - otherVals[windowEpoch];
      }
    }
  }

  return features;
}

/**
 * Create "all modalities above threshold" label for a future epoch.
 */
function createLabel(
  modalityTimeseries: Record<string, number[]>,
  futureEpoch: number,
  thresholdSigma: number
): number {
  const mods = Object.keys(modalityTimeseries);
  const aboveCount = mods.filter(
    (m) => modalityTimeseries[m][futureEpoch] >= thresholdSigma
  ).length;
  // Label = 1 if at least 2 modalities are above threshold (partial chord)
  // Use mods.length for full chord
  return aboveCount >= Math.max(2, Math.ceil(mods.length / 2)) ? 1 : 0;
}

/**
 * Standardize features (z-score) based on training set statistics.
 */
function standardizeFeatures(
  X: Record<string, number>[],
  stats?: { mean: Record<string, number>; std: Record<string, number> }
): { standardized: Record<string, number>[]; stats: { mean: Record<string, number>; std: Record<string, number> } } {
  const featureNames = Object.keys(X[0] || {});

  if (!stats) {
    stats = { mean: {}, std: {} };
    for (const name of featureNames) {
      const vals = X.map((row) => row[name] || 0);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
      stats.mean[name] = mean;
      stats.std[name] = std > 1e-12 ? std : 1;
    }
  }

  const standardized = X.map((row) => {
    const newRow: Record<string, number> = {};
    for (const name of featureNames) {
      newRow[name] = stats.std[name] > 0
        ? ((row[name] || 0) - stats.mean[name]) / stats.std[name]
        : 0;
    }
    return newRow;
  });

  return { standardized, stats };
}

/**
 * Simple logistic regression with gradient descent.
 * Returns weights and bias.
 */
function trainLogisticRegression(
  X: Record<string, number>[],
  y: number[],
  maxIter: number = 500,
  learningRate: number = 0.1,
  regularization: number = 0.01
): { weights: Record<string, number>; bias: number } {
  const featureNames = Object.keys(X[0] || {});
  const n = X.length;
  const weights: Record<string, number> = {};
  for (const name of featureNames) weights[name] = 0;
  let bias = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    const gradients: Record<string, number> = {};
    for (const name of featureNames) gradients[name] = 0;
    let gradBias = 0;

    for (let i = 0; i < n; i++) {
      let z = bias;
      for (const name of featureNames) z += weights[name] * (X[i][name] || 0);
      const pred = sigmoid(z);
      const error = pred - y[i];

      for (const name of featureNames) {
        gradients[name] += error * (X[i][name] || 0);
      }
      gradBias += error;
    }

    // Update with L2 regularization
    for (const name of featureNames) {
      weights[name] -= learningRate * (gradients[name] / n + regularization * weights[name]);
    }
    bias -= learningRate * (gradBias / n);
  }

  return { weights, bias };
}

/**
 * Leave-one-out cross-validation for logistic regression.
 */
function leaveOneOutCV(
  X: Record<string, number>[],
  y: number[]
): { predictions: number[]; weights: Record<string, number>; bias: number } {
  const n = X.length;
  const predictions: number[] = [];
  const featureNames = Object.keys(X[0] || {});
  let finalWeights: Record<string, number> = {};
  let finalBias = 0;

  for (let i = 0; i < n; i++) {
    const trainX = X.filter((_, idx) => idx !== i);
    const trainY = y.filter((_, idx) => idx !== i);
    const testX = X[i];

    const { standardized: sTrainX, stats } = standardizeFeatures(trainX);
    const sTestX: Record<string, number> = {};
    for (const name of featureNames) {
      sTestX[name] = stats.std[name] > 0
        ? ((testX[name] || 0) - stats.mean[name]) / stats.std[name]
        : 0;
    }

    const { weights, bias } = trainLogisticRegression(sTrainX, trainY);

    // On last iteration, save the model
    if (i === n - 1) {
      finalWeights = weights;
      finalBias = bias;
    }

    let z = bias;
    for (const name of featureNames) z += weights[name] * (sTestX[name] || 0);
    predictions.push(sigmoid(z) >= 0.5 ? 1 : 0);
  }

  // Train final model on all data for feature importance
  const { standardized: sAllX } = standardizeFeatures(X);
  const finalModel = trainLogisticRegression(sAllX, y);
  finalWeights = finalModel.weights;
  finalBias = finalModel.bias;

  return { predictions, weights: finalWeights, bias: finalBias };
}

/**
 * Compute approximate AUC using the ranking method (Wilcoxon-Mann-Whitney).
 */
function computeAUC(yTrue: number[], yScore: number[]): number {
  const n = yTrue.length;
  if (n === 0) return 0;

  const pairs: number[] = [];
  for (let i = 0; i < n; i++) {
    if (yTrue[i] === 1) pairs.push(yScore[i]);
  }
  const negatives: number[] = [];
  for (let i = 0; i < n; i++) {
    if (yTrue[i] === 0) negatives.push(yScore[i]);
  }

  let concordant = 0;
  let total = 0;
  for (const p of pairs) {
    for (const n of negatives) {
      if (p > n) concordant++;
      else if (p === n) concordant += 0.5;
      total++;
    }
  }

  return total > 0 ? concordant / total : 0;
}

/**
 * Run prediction window analysis on a single dyad.
 */
export function runPredictionWindowAnalysis(
  modalityTimeseries: Record<string, number[]>,
  epochMs: number,
  config?: Partial<PredictionConfig>
): PredictionResult {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const warnings: string[] = [];
  const mods = Object.keys(modalityTimeseries);
  const minLen = Math.min(...Object.values(modalityTimeseries).map((v) => v.length));

  if (minLen < 20) {
    warnings.push(`Very short timeseries (n=${minLen} epochs). Prediction results are unreliable.`);
  }

  const positiveCount = Math.ceil(mods.length / 2);

  // Build feature matrix and labels
  const featureRows: Record<string, number>[] = [];
  const labels: number[] = [];
  const rawPredictions: number[] = [];

  for (let t = 0; t < minLen - cfg.deltaT; t++) {
    const features = extractWindowFeatures(modalityTimeseries, t, cfg);
    const futureEpoch = t + cfg.deltaT;
    const label = createLabel(modalityTimeseries, futureEpoch, cfg.thresholdSigma);

    featureRows.push(features);
    labels.push(label);
    rawPredictions.push(
      mods.filter((m) => modalityTimeseries[m][futureEpoch] >= cfg.thresholdSigma).length
    );
  }

  const totalSamples = featureRows.length;
  const positiveSamples = labels.filter((l) => l === 1).length;
  const negativeSamples = totalSamples - positiveSamples;

  if (totalSamples < 10) {
    warnings.push("Insufficient samples for meaningful prediction analysis.");
    return emptyPredictionResult(cfg, epochMs, warnings);
  }

  if (positiveSamples < 3 || negativeSamples < 3) {
    warnings.push(
      `Severe class imbalance: ${positiveSamples} positive vs ${negativeSamples} negative samples. Results may be unreliable.`
    );
  }

  // Leave-one-out cross-validation
  const { predictions, weights } = leaveOneOutCV(featureRows, labels);

  // Compute raw scores for AUC (train on all, predict all)
  const { standardized: sAll } = standardizeFeatures(featureRows);
  const rawScores = sAll.map((row) => {
    let z = 0;
    for (const [name, w] of Object.entries(weights)) {
      z += w * (row[name] || 0);
    }
    return sigmoid(z);
  });

  // Confusion matrix
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < labels.length; i++) {
    if (predictions[i] === 1 && labels[i] === 1) tp++;
    else if (predictions[i] === 1 && labels[i] === 0) fp++;
    else if (predictions[i] === 0 && labels[i] === 0) tn++;
    else fn++;
  }

  const accuracy = totalSamples > 0 ? (tp + tn) / totalSamples : 0;
  const sensitivity = tp + fn > 0 ? tp / (tp + fn) : 0;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const f1 = precision + sensitivity > 0 ? (2 * precision * sensitivity) / (precision + sensitivity) : 0;
  const auc = computeAUC(labels, rawScores);

  // Feature importance: sorted by absolute weight
  const featureImportance = Object.entries(weights)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([name, weight]) => ({
      feature: name,
      weight: parseFloat(weight.toFixed(4)),
      description: describeFeature(name),
    }));

  // Summary
  const summaryParts: string[] = [];
  summaryParts.push(
    `Predicting ${cfg.deltaT}-epoch-ahead multimodal synchrony (≥${positiveCount} modalities above ${cfg.thresholdSigma}σ).`
  );
  summaryParts.push(
    `Accuracy=${(accuracy * 100).toFixed(1)}%, F1=${(f1 * 100).toFixed(1)}%, AUC=${auc.toFixed(3)}.`
  );

  if (featureImportance.length > 0) {
    const top3 = featureImportance.slice(0, 3);
    summaryParts.push(
      `Top 3 predictors: ${top3.map((f) => `${f.description} (${f.weight > 0 ? "+" : ""}${f.weight})`).join(", ")}.`
    );
  }

  if (auc > 0.7) {
    summaryParts.push(`Strong predictive signal detected (AUC > 0.7).`);
  } else if (auc > 0.6) {
    summaryParts.push(`Moderate predictive signal (AUC > 0.6).`);
  } else {
    summaryParts.push(`Weak or no predictive signal (AUC = ${auc.toFixed(3)}).`);
  }

  return {
    model: "Logistic Regression (LOO-CV)",
    deltaT: cfg.deltaT,
    deltaTSec: parseFloat(((cfg.deltaT * epochMs) / 1000).toFixed(1)),
    totalSamples,
    positiveSamples,
    negativeSamples,
    accuracy: parseFloat(accuracy.toFixed(4)),
    sensitivity: parseFloat(sensitivity.toFixed(4)),
    specificity: parseFloat(specificity.toFixed(4)),
    precision: parseFloat(precision.toFixed(4)),
    f1: parseFloat(f1.toFixed(4)),
    auc: parseFloat(auc.toFixed(4)),
    featureImportance,
    tp, fp, tn, fn,
    warnings,
    summary: summaryParts.join(" "),
  };
}

/**
 * Run prediction window analysis across multiple dyads.
 */
export function runMultiDyadPrediction(
  dyadData: { dyadId: string; modalityTimeseries: Record<string, number[]> }[],
  epochMs: number,
  config?: Partial<PredictionConfig>
): MultiDyadPredictionResult {
  const results = dyadData.map(({ dyadId, modalityTimeseries }) => ({
    dyadId,
    result: runPredictionWindowAnalysis(modalityTimeseries, epochMs, config),
  }));

  // Aggregate performance
  const accuracies = results.map((r) => r.result.accuracy);
  const f1s = results.map((r) => r.result.f1);
  const aucs = results.map((r) => r.result.auc);
  const sensitivities = results.map((r) => r.result.sensitivity);
  const precisions = results.map((r) => r.result.precision);

  // Find consistently important features
  const allFeatures = results.flatMap((r) => r.result.featureImportance.map((f) => f.feature));
  const featureFrequency: Record<string, { totalWeight: number; count: number }> = {};
  for (const result of results) {
    for (const f of result.result.featureImportance.slice(0, 5)) {
      if (!featureFrequency[f.feature]) featureFrequency[f.feature] = { totalWeight: 0, count: 0 };
      featureFrequency[f.feature].totalWeight += f.weight;
      featureFrequency[f.feature].count++;
    }
  }

  const consistentFeatures = Object.entries(featureFrequency)
    .filter(([_, v]) => v.count >= Math.ceil(results.length / 2))
    .map(([name, v]) => ({
      feature: name,
      meanWeight: parseFloat((v.totalWeight / v.count).toFixed(4)),
      frequency: v.count,
    }))
    .sort((a, b) => b.frequency - a.frequency);

  const warnings: string[] = [];
  if (results.length < 5) {
    warnings.push(`Few dyads (n=${results.length}). Aggregate statistics may not generalize.`);
  }

  const summaryParts: string[] = [];
  summaryParts.push(
    `Cross-dyad prediction: mean AUC=${(aucs.reduce((a, b) => a + b, 0) / aucs.length).toFixed(3)}, ` +
    `mean F1=${(f1s.reduce((a, b) => a + b, 0) / f1s.length).toFixed(3)} (n=${results.length} dyads).`
  );
  if (consistentFeatures.length > 0) {
    summaryParts.push(
      `Consistent predictors across dyads: ${consistentFeatures.slice(0, 3).map((f) => describeFeature(f.feature)).join(", ")}.`
    );
  }

  return {
    dyadPredictions: results,
    aggregatePerformance: {
      meanAccuracy: parseFloat((accuracies.reduce((a, b) => a + b, 0) / accuracies.length).toFixed(4)),
      meanF1: parseFloat((f1s.reduce((a, b) => a + b, 0) / f1s.length).toFixed(4)),
      meanAUC: parseFloat((aucs.reduce((a, b) => a + b, 0) / aucs.length).toFixed(4)),
      meanSensitivity: parseFloat((sensitivities.reduce((a, b) => a + b, 0) / sensitivities.length).toFixed(4)),
      meanPrecision: parseFloat((precisions.reduce((a, b) => a + b, 0) / precisions.length).toFixed(4)),
      consistentFeatures,
    },
    warnings,
    summary: summaryParts.join(" "),
  };
}

function describeFeature(featureName: string): string {
  if (featureName.endsWith("_value")) {
    const mod = featureName.replace("_value", "");
    return `${mod} sync level`;
  }
  if (featureName.endsWith("_changeRate")) {
    const mod = featureName.replace("_changeRate", "");
    return `${mod} sync change rate`;
  }
  if (featureName.endsWith("_acceleration")) {
    const mod = featureName.replace("_acceleration", "");
    return `${mod} sync acceleration`;
  }
  if (featureName.endsWith("_trendDev")) {
    const mod = featureName.replace("_trendDev", "");
    return `${mod} short-term trend deviation`;
  }
  if (featureName.includes("_minus_")) {
    const [mod1, mod2] = featureName.replace("_minus_", " vs ").split(" vs ");
    return `${mod1}−${mod2} difference`;
  }
  return featureName;
}

function emptyPredictionResult(
  config: PredictionConfig,
  epochMs: number,
  warnings: string[]
): PredictionResult {
  return {
    model: "Logistic Regression (LOO-CV)",
    deltaT: config.deltaT,
    deltaTSec: parseFloat(((config.deltaT * epochMs) / 1000).toFixed(1)),
    totalSamples: 0,
    positiveSamples: 0,
    negativeSamples: 0,
    accuracy: 0,
    sensitivity: 0,
    specificity: 0,
    precision: 0,
    f1: 0,
    auc: 0,
    featureImportance: [],
    tp: 0,
    fp: 0,
    tn: 0,
    fn: 0,
    warnings,
    summary: "Insufficient data for prediction analysis.",
  };
}
