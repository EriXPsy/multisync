/**
 * Sample datasets generated to faithfully represent real OSF synchrony research data.
 * Each dataset mirrors the data structures, sample rates, and signal characteristics
 * described in the cited open-science publications.
 *
 * IMPORTANT: These are synthetic timeseries generated to match published parameters,
 * NOT raw downloads from OSF. The original data can be accessed at the cited URLs.
 */

export interface SampleDataset {
  name: string;
  description: string;
  modality: "neural" | "behavioral" | "bio" | "psycho";
  osfUrl: string;
  citation: string;
  authors: string;
  year: number;
  streams: SampleStream[];
}

interface SampleStream {
  indexName: string;
  sampleRateHz: number;
  unit: string;
  durationMs: number;
  /** Generator produces { t, p1, p2 }[] */
  generateData: () => { t: number; p1: number; p2: number }[];
}

// ── Deterministic pseudo-random number generator (Mulberry32) ────────────────
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function generateSyncTimeseries(opts: {
  n: number;
  dt: number;
  baseFreqHz: number;
  couplingStrength: number;
  noiseLevel: number;
  seed: number;
  p1Offset?: number;
  p2Offset?: number;
}): { t: number; p1: number; p2: number }[] {
  const rng = mulberry32(opts.seed);
  const data: { t: number; p1: number; p2: number }[] = [];
  const { n, dt, baseFreqHz, couplingStrength, noiseLevel, p1Offset = 0, p2Offset = 0 } = opts;

  let phase1 = rng() * Math.PI * 2;
  let phase2 = rng() * Math.PI * 2;
  const omega = 2 * Math.PI * baseFreqHz;

  for (let i = 0; i < n; i++) {
    const t = i * dt;
    // Kuramoto-style coupling
    const coupling = couplingStrength * Math.sin(phase2 - phase1);
    phase1 += omega * (dt / 1000) + coupling * (dt / 1000) + (rng() - 0.5) * noiseLevel;
    phase2 += omega * (dt / 1000) - coupling * (dt / 1000) + (rng() - 0.5) * noiseLevel;

    data.push({
      t: Math.round(t),
      p1: parseFloat((Math.sin(phase1) + p1Offset + (rng() - 0.5) * noiseLevel * 0.3).toFixed(4)),
      p2: parseFloat((Math.sin(phase2) + p2Offset + (rng() - 0.5) * noiseLevel * 0.3).toFixed(4)),
    });
  }
  return data;
}

function generateIBI(opts: {
  durationMs: number;
  meanIBI_ms: number;
  variability: number;
  coupling: number;
  seed: number;
}): { t: number; p1: number; p2: number }[] {
  const rng = mulberry32(opts.seed);
  const data: { t: number; p1: number; p2: number }[] = [];
  let t1 = 0, t2 = 0;
  let ibi1 = opts.meanIBI_ms, ibi2 = opts.meanIBI_ms;
  const tMax = opts.durationMs;

  // Generate at 1 Hz effective resolution
  for (let t = 0; t < tMax; t += 1000) {
    // AR(1) process with coupling
    ibi1 = opts.meanIBI_ms + 0.7 * (ibi1 - opts.meanIBI_ms) + opts.coupling * (ibi2 - ibi1) + (rng() - 0.5) * opts.variability;
    ibi2 = opts.meanIBI_ms + 0.7 * (ibi2 - opts.meanIBI_ms) + opts.coupling * (ibi1 - ibi2) + (rng() - 0.5) * opts.variability;

    data.push({
      t,
      p1: parseFloat(Math.max(400, Math.min(1200, ibi1)).toFixed(1)),
      p2: parseFloat(Math.max(400, Math.min(1200, ibi2)).toFixed(1)),
    });
  }
  return data;
}

function generateEDA(opts: {
  durationMs: number;
  sampleRateHz: number;
  coupling: number;
  seed: number;
}): { t: number; p1: number; p2: number }[] {
  const rng = mulberry32(opts.seed);
  const dt = 1000 / opts.sampleRateHz;
  const n = Math.floor(opts.durationMs / dt);
  const data: { t: number; p1: number; p2: number }[] = [];
  let scl1 = 2 + rng() * 3;
  let scl2 = 2 + rng() * 3;

  for (let i = 0; i < n; i++) {
    // Tonic level with slow drift + coupled SCR events
    const drift1 = Math.sin(i * dt / 60000 * Math.PI) * 0.5;
    const drift2 = Math.sin((i * dt + 5000) / 60000 * Math.PI) * 0.5;

    // Random SCR events (shared + independent)
    const sharedEvent = rng() < 0.002 ? 1.5 : 0;
    const indep1 = rng() < 0.001 ? rng() * 1.0 : 0;
    const indep2 = rng() < 0.001 ? rng() * 1.0 : 0;

    scl1 = 0.999 * scl1 + 0.001 * (3 + drift1) + sharedEvent * opts.coupling + indep1;
    scl2 = 0.999 * scl2 + 0.001 * (3 + drift2) + sharedEvent * opts.coupling + indep2;

    if (i % 4 === 0) { // downsample to keep array manageable
      data.push({
        t: Math.round(i * dt),
        p1: parseFloat(scl1.toFixed(3)),
        p2: parseFloat(scl2.toFixed(3)),
      });
    }
  }
  return data;
}

function generateMEA(opts: {
  durationMs: number;
  fps: number;
  coupling: number;
  seed: number;
}): { t: number; p1: number; p2: number }[] {
  const rng = mulberry32(opts.seed);
  const dt = 1000 / opts.fps;
  const n = Math.min(Math.floor(opts.durationMs / dt), 9000); // cap at 5 min @ 30fps
  const data: { t: number; p1: number; p2: number }[] = [];
  let e1 = 0, e2 = 0;

  for (let i = 0; i < n; i++) {
    // Motion energy: bursts of activity with some coupling
    const burst = rng() < 0.05;
    const sharedBurst = rng() < 0.03;

    e1 = 0.85 * e1 + (burst || sharedBurst ? rng() * 50 : rng() * 2);
    e2 = 0.85 * e2 + (sharedBurst ? rng() * 50 : rng() < 0.05 ? rng() * 50 : rng() * 2);

    // Add coupling lag
    if (i > 5) {
      e2 += opts.coupling * data[i - 5].p1 * 0.05;
    }

    data.push({
      t: Math.round(i * dt),
      p1: parseFloat(Math.max(0, e1).toFixed(2)),
      p2: parseFloat(Math.max(0, e2).toFixed(2)),
    });
  }
  return data;
}

function generateLikertTimeseries(opts: {
  durationMs: number;
  intervalMs: number;
  scale: number;
  coupling: number;
  seed: number;
}): { t: number; p1: number; p2: number }[] {
  const rng = mulberry32(opts.seed);
  const n = Math.floor(opts.durationMs / opts.intervalMs);
  const data: { t: number; p1: number; p2: number }[] = [];
  let v1 = Math.floor(rng() * opts.scale) + 1;
  let v2 = Math.floor(rng() * opts.scale) + 1;

  for (let i = 0; i < n; i++) {
    // Random walk with coupling toward partner
    const delta1 = Math.round((rng() - 0.45) * 2 + opts.coupling * (v2 - v1));
    const delta2 = Math.round((rng() - 0.45) * 2 + opts.coupling * (v1 - v2));
    v1 = Math.max(1, Math.min(opts.scale, v1 + delta1));
    v2 = Math.max(1, Math.min(opts.scale, v2 + delta2));

    data.push({
      t: i * opts.intervalMs,
      p1: v1,
      p2: v2,
    });
  }
  return data;
}

// ── Sample Datasets ──────────────────────────────────────────────────────────

const DURATION_5MIN = 300_000;

export const SAMPLE_DATASETS: SampleDataset[] = [
  // ━━━ BIOSYNCHRONY ━━━
  {
    name: "Boukarras et al. – Dyadic Joint Action (HR & EDA)",
    description: "Heart rate (IBI) and electrodermal activity from dyads performing joint actions (tower-building). Data structure follows Empatica E4 format as described in the publication.",
    modality: "bio",
    osfUrl: "https://osf.io/preprints/osf/mr8j9",
    citation: "Boukarras, S., Placidi, V., Rossano, F., Era, V., Aglioti, S. M., & Candidi, M. (2025). Interpersonal physiological synchrony during dyadic joint action is increased by task novelty and reduced by social anxiety. Psychophysiology, 62, e70031.",
    authors: "Boukarras, Placidi, Rossano, Era, Aglioti & Candidi",
    year: 2025,
    streams: [
      {
        indexName: "heart_rate_ibi",
        sampleRateHz: 1,
        unit: "ms (IBI)",
        durationMs: DURATION_5MIN,
        generateData: () => generateIBI({ durationMs: DURATION_5MIN, meanIBI_ms: 780, variability: 80, coupling: 0.15, seed: 42 }),
      },
      {
        indexName: "eda_scl",
        sampleRateHz: 4,
        unit: "µS",
        durationMs: DURATION_5MIN,
        generateData: () => generateEDA({ durationMs: DURATION_5MIN, sampleRateHz: 4, coupling: 0.6, seed: 43 }),
      },
    ],
  },

  // ━━━ BEHAVIORAL SYNCHRONY ━━━
  {
    name: "Nelson et al. – Dyadic Rapport via Motion Energy Analysis",
    description: "Motion energy analysis (MEA) timeseries from videotaped dyadic interactions exploring the rapport–synchrony relationship. ROI-based frame-differencing at 30 fps.",
    modality: "behavioral",
    osfUrl: "https://osf.io/dyntp/",
    citation: "Nelson, A., Grahe, J., Ramseyer, F., & Serier, K. (2014). Psychological data from an exploration of the rapport/synchrony interplay using Motion Energy Analysis. Journal of Open Psychology Data, 2(1), e5.",
    authors: "Nelson, Grahe, Ramseyer & Serier",
    year: 2014,
    streams: [
      {
        indexName: "mea_upper_body",
        sampleRateHz: 30,
        unit: "px²/frame",
        durationMs: DURATION_5MIN,
        generateData: () => generateMEA({ durationMs: DURATION_5MIN, fps: 30, coupling: 0.3, seed: 100 }),
      },
      {
        indexName: "mea_head_region",
        sampleRateHz: 30,
        unit: "px²/frame",
        durationMs: DURATION_5MIN,
        generateData: () => generateMEA({ durationMs: DURATION_5MIN, fps: 30, coupling: 0.25, seed: 101 }),
      },
    ],
  },

  // ━━━ NEURAL SYNCHRONY ━━━
  {
    name: "Turk et al. – Brains in Sync (Mother-Infant EEG)",
    description: "Dual-EEG hyperscanning data from mother-infant free play. Alpha and theta band phase-locking values (PLV) computed from 10-20 montage channels (Fp1, Fp2, C3, C4, Cz, Pz).",
    modality: "neural",
    osfUrl: "https://osf.io/krdn5/",
    citation: "Turk, E., Endevelt-Shapira, Y., Feldman, R., van den Heuvel, M. I., & Levy, J. (2022). Brains in Sync: Practical guideline for parent-infant EEG during natural interaction. Frontiers in Psychology, 13, 833112.",
    authors: "Turk, Endevelt-Shapira, Feldman, van den Heuvel & Levy",
    year: 2022,
    streams: [
      {
        indexName: "eeg_alpha_plv",
        sampleRateHz: 10, // downsampled PLV timecourse
        unit: "PLV",
        durationMs: DURATION_5MIN,
        generateData: () => generateSyncTimeseries({
          n: 3000, dt: 100, baseFreqHz: 10, couplingStrength: 0.8,
          noiseLevel: 0.4, seed: 200, p1Offset: 0.3, p2Offset: 0.3,
        }).map(d => ({ t: d.t, p1: parseFloat(Math.abs(d.p1).toFixed(4)), p2: parseFloat(Math.abs(d.p2).toFixed(4)) })),
      },
      {
        indexName: "eeg_theta_plv",
        sampleRateHz: 10,
        unit: "PLV",
        durationMs: DURATION_5MIN,
        generateData: () => generateSyncTimeseries({
          n: 3000, dt: 100, baseFreqHz: 6, couplingStrength: 0.6,
          noiseLevel: 0.5, seed: 201, p1Offset: 0.2, p2Offset: 0.2,
        }).map(d => ({ t: d.t, p1: parseFloat(Math.abs(d.p1).toFixed(4)), p2: parseFloat(Math.abs(d.p2).toFixed(4)) })),
      },
    ],
  },

  // ━━━ PSYCHO-SYNCHRONY ━━━
  {
    name: "Baader et al. – IOS11 Relationship Closeness",
    description: "Extended Inclusion of Other in Self (IOS11) ratings from dyadic interaction experiments. 11-point Venn-diagram scale rated at 2-minute intervals, plus rapport and flow questionnaires.",
    modality: "psycho",
    osfUrl: "https://osf.io/5vrdz/",
    citation: "Baader, M., Starmer, C., Tufano, F., & Gächter, S. (2024). Introducing IOS11 as an extended interactive version of the 'Inclusion of Other in the Self' scale to estimate relationship closeness. Scientific Reports, 14, 8174.",
    authors: "Baader, Starmer, Tufano & Gächter",
    year: 2024,
    streams: [
      {
        indexName: "ios11_closeness",
        sampleRateHz: 0.0083, // 1 per 120s
        unit: "IOS (1-11)",
        durationMs: DURATION_5MIN,
        generateData: () => generateLikertTimeseries({ durationMs: DURATION_5MIN, intervalMs: 120_000, scale: 11, coupling: 0.3, seed: 300 }),
      },
      {
        indexName: "rapport_rating",
        sampleRateHz: 0.0083,
        unit: "Likert (1-7)",
        durationMs: DURATION_5MIN,
        generateData: () => generateLikertTimeseries({ durationMs: DURATION_5MIN, intervalMs: 120_000, scale: 7, coupling: 0.25, seed: 301 }),
      },
      {
        indexName: "flow_state_rating",
        sampleRateHz: 0.0056,
        unit: "Likert (1-7)",
        durationMs: DURATION_5MIN,
        generateData: () => generateLikertTimeseries({ durationMs: DURATION_5MIN, intervalMs: 180_000, scale: 7, coupling: 0.2, seed: 302 }),
      },
    ],
  },
];
