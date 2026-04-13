// Simulated multimodal synchrony data for demonstration
// Based on literature: neural ~ms, behavioral ~100ms-1s, bio ~1s, psycho ~minutes

export type ModalityType = "neural" | "behavioral" | "bio" | "psycho";

export interface SynchronyIndex {
  id: string;
  name: string;
  modality: ModalityType;
  description: string;
  nativeResolutionMs: number;
  unit: string;
  weight: number;
  method: string;
  reference: string;
}

export interface EpochDataPoint {
  epochIndex: number;
  timeStartMs: number;
  timeEndMs: number;
  value: number; // z-scored synchrony value
  rawValue: number;
  confidence: number;
}

export interface ModalityStream {
  modality: ModalityType;
  label: string;
  indices: SynchronyIndex[];
  compositeScore: EpochDataPoint[];
  color: string;
}

export interface TimelineConfig {
  commonEpochMs: number; // configurable common resolution
  totalDurationMs: number;
}

// Literature-grounded synchrony indices
export const SYNCHRONY_INDICES: SynchronyIndex[] = [
  // Neural synchrony indices
  {
    id: "eeg-alpha",
    name: "Alpha-band IBS",
    modality: "neural",
    description: "Inter-brain synchrony in alpha (8-12 Hz) band via EEG hyperscanning. Reflects shared attention and coordinated task engagement.",
    nativeResolutionMs: 4, // ~250 Hz EEG
    unit: "PLV",
    weight: 0.35,
    method: "Phase Locking Value (PLV) or Circular Correlation",
    reference: "Burns et al., 2025; Dikker et al., 2017",
  },
  {
    id: "eeg-theta",
    name: "Theta-band IBS",
    modality: "neural",
    description: "Inter-brain synchrony in theta (4-7 Hz) band. Associated with memory encoding and social prediction.",
    nativeResolutionMs: 4,
    unit: "PLV",
    weight: 0.25,
    method: "Phase Locking Value (PLV)",
    reference: "Goldstein et al., 2018",
  },
  {
    id: "fnirs-pfc",
    name: "Prefrontal fNIRS Sync",
    modality: "neural",
    description: "Prefrontal cortex inter-brain synchrony via fNIRS. Tracks social cognition and joint decision-making.",
    nativeResolutionMs: 100, // ~10 Hz fNIRS
    unit: "WTC",
    weight: 0.25,
    method: "Wavelet Transform Coherence (WTC)",
    reference: "Carollo et al., 2025; Cui et al., 2012",
  },
  {
    id: "fnirs-tpj",
    name: "TPJ fNIRS Sync",
    modality: "neural",
    description: "Temporoparietal junction inter-brain sync. Related to mentalizing and perspective-taking.",
    nativeResolutionMs: 100,
    unit: "WTC",
    weight: 0.15,
    method: "Wavelet Transform Coherence (WTC)",
    reference: "Schäfer et al., 2026",
  },

  // Behavioral synchrony indices
  {
    id: "eye-gaze",
    name: "Gaze Coordination",
    modality: "behavioral",
    description: "Temporal alignment of gaze patterns. Includes mutual gaze, joint attention, and gaze following.",
    nativeResolutionMs: 33, // ~30 Hz eye tracker
    unit: "r_wcc",
    weight: 0.25,
    method: "Windowed Cross-Correlation (WCC) of gaze x,y coordinates",
    reference: "Behrens et al., 2020; Prochazkova et al., 2018",
  },
  {
    id: "head-movement",
    name: "Head Movement Sync",
    modality: "behavioral",
    description: "Coordination of head orientation and nodding patterns during interaction.",
    nativeResolutionMs: 33,
    unit: "r_wcc",
    weight: 0.20,
    method: "WCC of 3-axis head rotation (pitch, yaw, roll)",
    reference: "Ramseyer & Tschacher, 2011",
  },
  {
    id: "gesture",
    name: "Gesture Synchrony",
    modality: "behavioral",
    description: "Temporal alignment of hand/arm gestures. Coded via motion energy analysis or pose estimation.",
    nativeResolutionMs: 33,
    unit: "r_wcc",
    weight: 0.20,
    method: "Motion Energy Analysis (MEA) or pose-estimation WCC",
    reference: "Koul et al., 2023; Yozevitch et al., 2023",
  },
  {
    id: "facial-expression",
    name: "Facial Expression Sync",
    modality: "behavioral",
    description: "Mimicry and temporal alignment of facial action units (AUs). Reflects emotional contagion.",
    nativeResolutionMs: 33,
    unit: "r_wcc",
    weight: 0.20,
    method: "WCC of AU intensities from FACS coding or ML-based AU detection",
    reference: "Chartrand & Bargh, 1999; Behrens et al., 2020",
  },
  {
    id: "posture",
    name: "Postural Sway Sync",
    modality: "behavioral",
    description: "Body posture alignment and postural sway coordination during interaction.",
    nativeResolutionMs: 100,
    unit: "r_wcc",
    weight: 0.15,
    method: "WCC of center-of-pressure or skeletal trunk position",
    reference: "Ramseyer & Tschacher, 2011",
  },

  // Biosynchrony indices
  {
    id: "heart-rate",
    name: "Heart Rate Sync",
    modality: "bio",
    description: "Temporal coordination of cardiac inter-beat intervals (IBI). Marker of shared arousal and co-regulation.",
    nativeResolutionMs: 1000, // ~1 Hz effective IBI rate
    unit: "r_wcc",
    weight: 0.30,
    method: "WCC of IBI series (window: 15s, lag: ±5s per Behrens et al.)",
    reference: "Burns et al., 2025; Feldman et al., 2011",
  },
  {
    id: "eda",
    name: "EDA Sync",
    modality: "bio",
    description: "Skin conductance level alignment. Reflects shared sympathetic nervous system activation.",
    nativeResolutionMs: 250, // ~4 Hz SCL
    unit: "r_wcc",
    weight: 0.25,
    method: "WCC of SCL (window: 30s, lag: ±7s per Behrens et al.)",
    reference: "Behrens et al., 2020; Ohayon et al., 2026",
  },
  {
    id: "respiration",
    name: "Respiratory Sync",
    modality: "bio",
    description: "Breathing rhythm coordination. Can emerge during joint vocalization or shared physical tasks.",
    nativeResolutionMs: 500,
    unit: "r_wcc",
    weight: 0.20,
    method: "WCC of respiratory belt signal or estimated respiratory rate",
    reference: "Müller & Lindenberger, 2011",
  },
  {
    id: "pupil",
    name: "Pupil Dilation Sync",
    modality: "bio",
    description: "Coordinated pupil size changes. Reflects shared cognitive load and emotional arousal.",
    nativeResolutionMs: 33,
    unit: "r_wcc",
    weight: 0.25,
    method: "WCC of pupil diameter (window: 8s, lag: ±3s per Behrens et al.)",
    reference: "Kret & de Dreu, 2017; Behrens et al., 2020",
  },

  // Psycho-synchrony indices
  {
    id: "ios",
    name: "IOS Closeness",
    modality: "psycho",
    description: "Inclusion of Other in Self scale. Discrete rating (1-7 or 1-11 for IOS11) of perceived interpersonal closeness.",
    nativeResolutionMs: 120000, // rated every ~2 minutes
    unit: "IOS",
    weight: 0.40,
    method: "IOS11 Venn diagram rating (Baader et al., 2024) at epoch intervals",
    reference: "Aron et al., 1992; Baader et al., 2024",
  },
  {
    id: "rapport",
    name: "Perceived Rapport",
    modality: "psycho",
    description: "Self-reported rapport and connection quality during interaction episodes.",
    nativeResolutionMs: 120000,
    unit: "Likert",
    weight: 0.30,
    method: "Continuous slider or discrete Likert scale at epoch boundaries",
    reference: "Bernieri, 1988; Tickle-Degnen & Rosenthal, 1990",
  },
  {
    id: "flow-state",
    name: "Shared Flow State",
    modality: "psycho",
    description: "Joint experience of psychological flow during collaborative tasks.",
    nativeResolutionMs: 180000,
    unit: "Likert",
    weight: 0.30,
    method: "Post-epoch flow questionnaire or continuous dial rating",
    reference: "Csikszentmihalyi, 1990; Magyaródi et al., 2017",
  },
];

// Generate simulated epoch data for demonstration
function generateEpochData(
  numEpochs: number,
  epochMs: number,
  modality: ModalityType,
  seed: number = 0,
  onsetConfig?: Record<ModalityType, number>
): EpochDataPoint[] {
  const data: EpochDataPoint[] = [];
  
  // Configurable cascade onset — no assumed "correct" order
  const defaultOnsets: Record<ModalityType, number> = {
    neural: 1,
    behavioral: 3,
    bio: 5,
    psycho: 8,
  };
  const onsetEpoch = (onsetConfig || defaultOnsets)[modality];

  const peakEpoch = Math.floor(numEpochs * 0.6);
  
  for (let i = 0; i < numEpochs; i++) {
    const t = i / numEpochs;
    const onset = 1 / (1 + Math.exp(-2 * (i - onsetEpoch)));
    const peak = Math.exp(-0.5 * Math.pow((i - peakEpoch) / (numEpochs * 0.25), 2));
    const noise = (Math.sin(seed * 17 + i * 3.7) * 0.5 + Math.cos(seed * 11 + i * 2.3) * 0.3) * 0.15;
    
    const rawValue = onset * (0.3 + 0.7 * peak) + noise;
    const value = (rawValue - 0.5) / 0.25;
    
    data.push({
      epochIndex: i,
      timeStartMs: i * epochMs,
      timeEndMs: (i + 1) * epochMs,
      value: parseFloat(value.toFixed(3)),
      rawValue: parseFloat(Math.max(0, Math.min(1, rawValue)).toFixed(3)),
      confidence: parseFloat((0.7 + 0.3 * onset * (1 - Math.abs(noise))).toFixed(2)),
    });
  }
  
  return data;
}

export function generateDemoData(config: TimelineConfig): ModalityStream[] {
  const numEpochs = Math.floor(config.totalDurationMs / config.commonEpochMs);
  
  return [
    {
      modality: "neural",
      label: "Neural Synchrony",
      indices: SYNCHRONY_INDICES.filter(i => i.modality === "neural"),
      compositeScore: generateEpochData(numEpochs, config.commonEpochMs, "neural", 1),
      color: "hsl(var(--neural))",
    },
    {
      modality: "behavioral",
      label: "Behavioral Synchrony",
      indices: SYNCHRONY_INDICES.filter(i => i.modality === "behavioral"),
      compositeScore: generateEpochData(numEpochs, config.commonEpochMs, "behavioral", 2),
      color: "hsl(var(--behavioral))",
    },
    {
      modality: "bio",
      label: "Biosynchrony",
      indices: SYNCHRONY_INDICES.filter(i => i.modality === "bio"),
      compositeScore: generateEpochData(numEpochs, config.commonEpochMs, "bio", 3),
      color: "hsl(var(--bio))",
    },
    {
      modality: "psycho",
      label: "Psycho-synchrony",
      indices: SYNCHRONY_INDICES.filter(i => i.modality === "psycho"),
      compositeScore: generateEpochData(numEpochs, config.commonEpochMs, "psycho", 4),
      color: "hsl(var(--psycho))",
    },
  ];
}

export interface WCCParameters {
  windowSizeMs: number;
  maxLagMs: number;
  windowIncrementMs: number;
  lagIncrementMs: number;
}

export const DEFAULT_WCC_PARAMS: Record<string, WCCParameters> = {
  "eeg-alpha": { windowSizeMs: 2000, maxLagMs: 500, windowIncrementMs: 100, lagIncrementMs: 10 },
  "eeg-theta": { windowSizeMs: 2000, maxLagMs: 500, windowIncrementMs: 100, lagIncrementMs: 10 },
  "fnirs-pfc": { windowSizeMs: 10000, maxLagMs: 3000, windowIncrementMs: 1000, lagIncrementMs: 100 },
  "fnirs-tpj": { windowSizeMs: 10000, maxLagMs: 3000, windowIncrementMs: 1000, lagIncrementMs: 100 },
  "eye-gaze": { windowSizeMs: 5000, maxLagMs: 2000, windowIncrementMs: 500, lagIncrementMs: 33 },
  "head-movement": { windowSizeMs: 5000, maxLagMs: 2000, windowIncrementMs: 500, lagIncrementMs: 33 },
  "gesture": { windowSizeMs: 8000, maxLagMs: 3000, windowIncrementMs: 500, lagIncrementMs: 33 },
  "facial-expression": { windowSizeMs: 5000, maxLagMs: 2000, windowIncrementMs: 500, lagIncrementMs: 33 },
  "posture": { windowSizeMs: 10000, maxLagMs: 3000, windowIncrementMs: 1000, lagIncrementMs: 100 },
  "heart-rate": { windowSizeMs: 15000, maxLagMs: 5000, windowIncrementMs: 1000, lagIncrementMs: 250 },
  "eda": { windowSizeMs: 30000, maxLagMs: 7000, windowIncrementMs: 2000, lagIncrementMs: 250 },
  "respiration": { windowSizeMs: 20000, maxLagMs: 5000, windowIncrementMs: 2000, lagIncrementMs: 500 },
  "pupil": { windowSizeMs: 8000, maxLagMs: 3000, windowIncrementMs: 500, lagIncrementMs: 33 },
  "ios": { windowSizeMs: 120000, maxLagMs: 0, windowIncrementMs: 120000, lagIncrementMs: 0 },
  "rapport": { windowSizeMs: 120000, maxLagMs: 0, windowIncrementMs: 120000, lagIncrementMs: 0 },
  "flow-state": { windowSizeMs: 180000, maxLagMs: 0, windowIncrementMs: 180000, lagIncrementMs: 0 },
};

export const REFERENCES = [
  { key: "Burns2025", citation: "Burns, A., Tomashin, A., Atia, B., Gilboa, A., Cohen, S., & Gordon, I. (2025). Flexible multimodal synchrony during joint drumming reflects interaction demands and social connection. Research Square.", category: "multimodal" },
  { key: "Gordon2025", citation: "Gordon, I., Tomashin, A., & Mayo, O. (2025). A theory of flexible multimodal synchrony. Psychological Review, 132(3), 680-718.", category: "theory" },
  { key: "Behrens2020", citation: "Behrens, F., Moulder, R. G., Boker, S. M., & Kret, M. E. (2020). Quantifying physiological synchrony through windowed cross-correlation analysis. bioRxiv.", category: "methods" },
  { key: "Tomashin2024", citation: "Tomashin, A., Gordon, I., & Wallot, S. (2024). Multimodal interpersonal synchrony: Systematic review and meta-analysis.", category: "multimodal" },
  { key: "Chidichimo2026", citation: "Chidichimo, E., et al. (2026). Towards an informational account of interpersonal coordination. Nature Reviews Neuroscience, 27(2), 121-137.", category: "theory" },
  { key: "Baader2024", citation: "Baader, M., Starmer, C., Tufano, F., & Gächter, S. (2024). Introducing IOS11 as an extended interactive version of the 'Inclusion of Other in the Self' scale. Scientific Reports.", category: "psycho" },
  { key: "Kappel2025", citation: "Kappel, S. L., Jørgensen, A. N., & Kidmose, P. (2025). Temporal synchronization of multimodal hyperscanning recordings: Challenges, methodologies, and best practices. EMBC.", category: "methods" },
  { key: "Boker2002", citation: "Boker, S. M., Xu, M., Rotondo, J. L., & King, K. (2002). Windowed cross-correlation and peak picking for the analysis of variability in the association between behavioral time series. Psychological Methods, 7(3), 338.", category: "methods" },
  { key: "Dikker2017", citation: "Dikker, S., et al. (2017). Brain-to-brain synchrony tracks real-world dynamic group interactions in the classroom. Current Biology, 27(9), 1375-1380.", category: "neural" },
  { key: "Feldman2011", citation: "Feldman, R., Magori-Cohen, R., Galili, G., Singer, M., & Louzoun, Y. (2011). Mother and infant coordinate heart rhythms through episodes of interaction synchrony. Infant Behavior and Development.", category: "bio" },
  { key: "Ramseyer2011", citation: "Ramseyer, F. & Tschacher, W. (2011). Nonverbal synchrony in psychotherapy. Journal of Consulting and Clinical Psychology, 79(3), 284.", category: "behavioral" },
  { key: "Koul2023", citation: "Koul, A., Ahmar, D., Iannetti, G. D., & Novembre, G. (2023). Spontaneous dyadic behavior predicts the emergence of interpersonal neural synchrony. NeuroImage, 277, 120233.", category: "multimodal" },
  { key: "Yozevitch2023", citation: "Yozevitch, R., Dahan, A., Seada, T., Appel, D., & Gvirts, H. (2023). Classifying interpersonal synchronization states using a data-driven approach. Scientific Reports.", category: "behavioral" },
  { key: "Meier2021", citation: "Meier, D. & Tschacher, W. (2021). Beyond dyadic coupling: The method of multivariate surrogate synchrony (mv-SUSY). Entropy, 23, 1385.", category: "methods" },
  { key: "Hudson2022", citation: "Hudson, D., Wiltshire, T. J., & Atzmueller, M. (2022). multiSyncPy: A Python package for assessing multivariate coordination dynamics. Behavior Research Methods, 55, 932-962.", category: "methods" },
  { key: "Ohayon2026", citation: "Ohayon, S., Erez, C., & Gordon, I. (2026). Demographic differences are associated with temporal variation in cardiac and electrodermal interpersonal synchrony. Scientific Reports.", category: "bio" },
  { key: "Aron1992", citation: "Aron, A., Aron, E. N., & Smollan, D. (1992). Inclusion of Other in the Self Scale and the structure of interpersonal closeness. Journal of Personality and Social Psychology, 63(4), 596.", category: "psycho" },
  { key: "Sun2024", citation: "Sun, Y., Day, S., Gilbert, T., Maria, B., Hamilton, A. F. C., & Ward, J. A. (2024). Quantification and visualisation of interpersonal synchrony using wearable sensors and time series similarity analysis. TechRxiv.", category: "methods" },
  { key: "Quan2025", citation: "Quan, J., Miyake, Y., & Nozawa, T. (2025). Incorporating multimodal directional interpersonal synchrony into empathetic response generation. Sensors, 25(2), 434.", category: "multimodal" },
  { key: "Chartrand1999", citation: "Chartrand, T. L. & Bargh, J. A. (1999). The chameleon effect: The perception–behavior link and social interaction. Journal of Personality and Social Psychology, 76(6), 893.", category: "behavioral" },
];
