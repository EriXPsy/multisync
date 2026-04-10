// Smart CSV parser with automatic column detection for dyadic synchrony data
// Supports formats from OSF, Physionet, Empatica E4, rMEA, NIRx fNIRS, and more

export type ParsedCSV = {
  headers: string[];
  rows: (number | string)[][];
  preview: (number | string)[][];
  rawLines: string[];
  formatHint?: FileFormatHint;
};

export type FileFormatHint = {
  format: "standard" | "empatica" | "rmea" | "nirx" | "ibi" | "physionet" | "biopac" | "opensignals" | "edf";
  description: string;
  detectedSampleRate?: number;
  detectedStartTime?: number; // unix epoch or relative
  metadataRows?: number; // rows consumed as metadata
};

export type ColumnDetection = {
  timestampCols: number[];
  person1Cols: number[];
  person2Cols: number[];
  signalCols: number[];
  unknownCols: number[];
  confidence: number;
  suggestions: string[];
};

// ──────────────────────────────────────────────────────
// TIMESTAMP detection patterns
// ──────────────────────────────────────────────────────
const TIMESTAMP_PATTERNS = [
  /^(time|timestamp|t|ms|seconds?|secs?|epoch|frame|sample_?n?|onset)$/i,
  /time_?(stamp)?/i,
  /^t_?\d*/i,
  /elapsed/i,
  /^frame_?/i,
  /^(unix|utc|clock)/i,
  /^(onset|offset|start_?time|end_?time)/i,
  /^(trigger_?time|event_?time|marker_?time)/i,
  /^sample_?(num|number|idx|index)?$/i,
  /^(rec_?time|recording_?time)/i,
  /timestamp.*\[.*\]/i, // e.g. "timestamp [s]"
];

// ──────────────────────────────────────────────────────
// PERSON / PARTICIPANT detection patterns
// Expanded with real-world OSF dataset naming conventions
// ──────────────────────────────────────────────────────
const PERSON1_PATTERNS = [
  /^(p1|person_?1|subj(ect)?_?1|participant_?1|child|speaker_?1|s1|dyad_?1)$/i,
  /person_?1/i,
  /participant_?a/i,
  /^(left|first|mother|parent|therapist|teacher|leader|caregiver|adult|experimenter)/i,
  /_p1$/i, /_s1$/i, /_person1$/i,
  /^(mom|mum|father|dad)_/i,
  /_?(mother|parent|therapist|teacher|leader|caregiver|adult|experimenter)$/i,
  /^(roi|ch|channel).*_?(sub|subj|part)_?1/i,
  /^(sub|subj|part)_?0?1_/i,
  /_dyad_?member_?1/i,
  /^interactor_?1/i,
  /^member_?a/i,
];

const PERSON2_PATTERNS = [
  /^(p2|person_?2|subj(ect)?_?2|participant_?2|partner|speaker_?2|s2|dyad_?2)$/i,
  /person_?2/i,
  /participant_?b/i,
  /^(right|second|infant|client|student|follower|baby|toddler|peer)/i,
  /_p2$/i, /_s2$/i, /_person2$/i,
  /^(infant|baby|toddler|child|kid)_/i,
  /_?(infant|client|student|follower|baby|toddler|peer)$/i,
  /^(roi|ch|channel).*_?(sub|subj|part)_?2/i,
  /^(sub|subj|part)_?0?2_/i,
  /_dyad_?member_?2/i,
  /^interactor_?2/i,
  /^member_?b/i,
];

// ──────────────────────────────────────────────────────
// SIGNAL type detection — much expanded from OSF datasets
// ──────────────────────────────────────────────────────
const SIGNAL_PATTERNS = [
  // EEG
  /^(eeg|fnirs|nirs|fmri)/i,
  /(alpha|beta|theta|gamma|delta|mu)(_?power|_?band)?/i,
  /^(fp[12z]|f[z34578]|c[z34]|p[z34578]|o[12z]|t[3-8]|af[34578]|fc[1-6z]|cp[1-6z]|po[34578z]|tp[789])/i, // 10-20 EEG channels
  /^(plv|pli|coherence|imaginary_?coh|envelope_?corr|power_?corr)/i,
  // fNIRS
  /(hbo|hbr|hbt|oxy|deoxy|total_?hb)/i,
  /^(s\d+_d\d+|source_?\d+.*det)/i, // source-detector pairs
  /^(roi|region)_?\d+/i,
  /^(ch|channel)_?\d+/i,
  // Cardiac / HRV
  /^(hr|heart_?rate|bpm|ibi|rr_?interval|nn_?interval)/i,
  /^(hrv|rmssd|sdnn|pnn50|lf|hf|lf_?hf|vlf)/i,
  /^(ecg|ekg|ppg|bvp|pulse)/i,
  /^(systolic|diastolic|blood_?pressure|bp)/i,
  /^(pep|lvet|co|cardiac_?output|stroke_?volume)/i,
  // Electrodermal
  /^(eda|scl|scr|gsr|skin_?cond|galvanic)/i,
  /^(tonic|phasic|eda_?tonic|eda_?phasic)/i,
  // Respiration
  /^(resp|respiration|breathing|br|rsp)/i,
  /^(resp_?rate|breathing_?rate|insp|expir)/i,
  // EMG
  /^(emg|electromyog)/i,
  // Temperature
  /^(temp|temperature|skt|skin_?temp)/i,
  // Motion / Kinematics
  /^(acc|accel|accelero)/i,
  /(x|y|z)_?(coord|pos|vel|acc|axis)/i,
  /(pitch|yaw|roll|heading)/i,
  /^(gyro|magnetom)/i,
  /^(mea|motion_?energy)/i,
  /^(velocity|speed|displacement)/i,
  // Eye tracking / Gaze
  /^(gaze|pupil|fixation|saccade)/i,
  /^(eye|blink|dwell)/i,
  /^(aoi|area_?of_?interest)/i,
  // Facial / AU
  /^(au|action_?unit)/i,
  /^(facs|facial)/i,
  /^(valence|arousal|dominance)/i,
  /^(smile|frown|brow)/i,
  // Audio / Vocal
  /^(f0|pitch|loudness|intensity|formant|mfcc)/i,
  /^(speaking|silence|turn_?taking|overlap)/i,
  /^(prosody|fundamental_?freq)/i,
  // Synchrony indices
  /^(sync|synchrony|coupling|coordination)/i,
  /^(cross_?corr|ccf|wcc|dtw|recurrence)/i,
  /^(coherence|granger|transfer_?entropy)/i,
  // Generic
  /^(signal|value|amplitude|power|magnitude)/i,
  /^(raw|filtered|processed|clean)/i,
];

// ──────────────────────────────────────────────────────
// Empatica E4 format detection
// First row = unix timestamp, second row = sample rate
// ──────────────────────────────────────────────────────
function detectEmpaticaFormat(rawLines: string[]): FileFormatHint | null {
  if (rawLines.length < 3) return null;
  const line1 = rawLines[0].trim().split(",");
  const line2 = rawLines[1].trim().split(",");
  
  // Empatica: row 1 has a single large number (unix timestamp), row 2 has sample rate
  if (line1.length <= 2 && line2.length <= 2) {
    const ts = parseFloat(line1[0]);
    const sr = parseFloat(line2[0]);
    if (ts > 1e9 && ts < 2e10 && sr > 0 && sr <= 1000) {
      return {
        format: "empatica",
        description: `Empatica E4 format: start=${new Date(ts * 1000).toISOString()}, rate=${sr}Hz`,
        detectedSampleRate: sr,
        detectedStartTime: ts,
        metadataRows: 2,
      };
    }
  }
  return null;
}

// ──────────────────────────────────────────────────────
// OpenSignals (BITalino) format detection
// Lines starting with # are header metadata
// ──────────────────────────────────────────────────────
function detectOpenSignalsFormat(rawLines: string[]): FileFormatHint | null {
  if (rawLines.length < 2) return null;
  if (rawLines[0].startsWith("# OpenSignals") || rawLines[0].startsWith("#OpenSignals")) {
    let metadataRows = 0;
    for (const line of rawLines) {
      if (line.startsWith("#")) metadataRows++;
      else break;
    }
    return {
      format: "opensignals",
      description: `OpenSignals/BITalino format: ${metadataRows} metadata rows`,
      metadataRows,
    };
  }
  return null;
}

// ──────────────────────────────────────────────────────
// Biopac AcqKnowledge format detection
// ──────────────────────────────────────────────────────
function detectBiopacFormat(rawLines: string[]): FileFormatHint | null {
  if (rawLines.length < 3) return null;
  const headerLower = rawLines[0].toLowerCase();
  if (headerLower.includes("acqknowledge") || headerLower.includes("biopac") || 
      headerLower.includes("mp150") || headerLower.includes("mp160")) {
    let metadataRows = 0;
    for (const line of rawLines) {
      const firstVal = line.split(/[,\t]/)[0].trim();
      if (isNaN(parseFloat(firstVal)) && metadataRows < 10) metadataRows++;
      else break;
    }
    return {
      format: "biopac",
      description: `Biopac AcqKnowledge export: ${metadataRows} header rows`,
      metadataRows: Math.max(metadataRows - 1, 0), // keep last text row as header
    };
  }
  return null;
}

// ──────────────────────────────────────────────────────
// IBI (Inter-Beat Interval) format
// Two columns: relative time, interval duration
// ──────────────────────────────────────────────────────
function detectIBIFormat(rawLines: string[], headers: string[]): FileFormatHint | null {
  if (headers.length !== 2) return null;
  const h = headers.join(" ").toLowerCase();
  if (/ibi|inter.?beat|rr.?interval/.test(h) || 
      (headers.length === 2 && rawLines.length > 5)) {
    // Check if two numeric columns with small values
    const testRows = rawLines.slice(1, 6);
    const allTwoCol = testRows.every(line => {
      const parts = line.split(/[,\t]/).map(s => parseFloat(s.trim()));
      return parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1]) && parts[1] > 0 && parts[1] < 5;
    });
    if (allTwoCol) {
      return {
        format: "ibi",
        description: "IBI (Inter-Beat Interval) format: time + interval duration",
      };
    }
  }
  return null;
}

// ──────────────────────────────────────────────────────
// rMEA motion energy format
// Plain text with two columns (s1, s2), no headers typically
// ──────────────────────────────────────────────────────
function detectRMEAFormat(rawLines: string[], fileName: string): FileFormatHint | null {
  if (!fileName) return null;
  // rMEA files are often .txt with dyad ID in filename
  if (!/\.(txt|csv)$/i.test(fileName)) return null;
  
  // Check: all rows have exactly 1-2 tab/comma separated numbers, no header
  const testRows = rawLines.slice(0, 10);
  const allNumeric = testRows.every(line => {
    const parts = line.trim().split(/[,\t\s]+/);
    return parts.length >= 1 && parts.length <= 3 && parts.every(p => !isNaN(parseFloat(p)));
  });
  
  if (allNumeric && rawLines.length > 20) {
    const cols = rawLines[0].trim().split(/[,\t\s]+/).length;
    if (cols === 2) {
      return {
        format: "rmea",
        description: "rMEA-style motion energy: 2-column (Person1, Person2), no headers",
        detectedSampleRate: 25, // rMEA default
      };
    }
  }
  return null;
}

// ──────────────────────────────────────────────────────
// Physionet format detection
// Often has metadata comments at top
// ──────────────────────────────────────────────────────
function detectPhysionetFormat(rawLines: string[]): FileFormatHint | null {
  if (rawLines.length < 2) return null;
  if (rawLines[0].startsWith("'Elapsed time'") || rawLines[0].includes("PhysioNet")) {
    return {
      format: "physionet",
      description: "PhysioNet data export format",
    };
  }
  return null;
}

// ──────────────────────────────────────────────────────
// Main format detection orchestrator
// ──────────────────────────────────────────────────────
export function detectFileFormat(rawLines: string[], fileName: string = ""): FileFormatHint | null {
  return (
    detectEmpaticaFormat(rawLines) ||
    detectOpenSignalsFormat(rawLines) ||
    detectBiopacFormat(rawLines) ||
    detectPhysionetFormat(rawLines) ||
    null
    // IBI and rMEA require parsed headers, handled in parseCSV
  );
}

// ──────────────────────────────────────────────────────
// Core CSV/TSV parser with special format handling
// ──────────────────────────────────────────────────────
export function parseCSV(text: string, fileName: string = ""): ParsedCSV {
  const rawLines = text.trim().split(/\r?\n/);
  
  // Detect special formats first
  let formatHint = detectFileFormat(rawLines, fileName);
  let startRow = 0;
  
  // Handle metadata rows for special formats
  if (formatHint?.metadataRows) {
    startRow = formatHint.metadataRows;
  }

  // Skip comment lines (# or %)
  while (startRow < rawLines.length && /^[#%]/.test(rawLines[startRow].trim())) {
    startRow++;
  }

  const workingLines = rawLines.slice(startRow);
  if (workingLines.length === 0) {
    return { headers: [], rows: [], preview: [], rawLines, formatHint: formatHint || undefined };
  }

  // Detect delimiter: tab > semicolon > comma
  const firstLine = workingLines[0];
  const delimiter = firstLine.includes("\t") ? "\t" : firstLine.includes(";") ? ";" : ",";
  
  // Check if first row is headers or data (for headerless files like rMEA)
  const firstRowParts = firstLine.split(delimiter).map(v => v.trim().replace(/^"|"$/g, ""));
  const firstRowAllNumeric = firstRowParts.every(v => !isNaN(parseFloat(v)) && v.length > 0);
  
  // Check for rMEA format (headerless numeric)
  if (firstRowAllNumeric && !formatHint) {
    const rmeaHint = detectRMEAFormat(workingLines, fileName);
    if (rmeaHint) formatHint = rmeaHint;
  }

  let headers: string[];
  let dataStartIdx: number;

  if (firstRowAllNumeric) {
    // Generate synthetic headers
    const numCols = firstRowParts.length;
    if (numCols === 1) headers = ["value"];
    else if (numCols === 2) headers = ["person1_mea", "person2_mea"];
    else if (numCols === 3) headers = ["time", "person1", "person2"];
    else headers = firstRowParts.map((_, i) => `col_${i + 1}`);
    dataStartIdx = 0;
  } else {
    headers = firstRowParts;
    dataStartIdx = 1;
  }

  // For Empatica format: single-value column, generate timestamp from sample rate
  if (formatHint?.format === "empatica" && formatHint.detectedSampleRate) {
    const sr = formatHint.detectedSampleRate;
    const startTime = formatHint.detectedStartTime || 0;
    headers = ["timestamp_s", ...headers.filter(h => h.toLowerCase() !== "timestamp")];
    
    const rows = workingLines.slice(dataStartIdx).filter(l => l.trim()).map((line, i) => {
      const vals = line.split(delimiter).map(v => {
        const trimmed = v.trim().replace(/^"|"$/g, "");
        const n = parseFloat(trimmed);
        return isNaN(n) ? trimmed : n;
      });
      return [i / sr, ...vals] as (number | string)[];
    });

    return {
      headers,
      rows,
      preview: rows.slice(0, 8),
      rawLines,
      formatHint: formatHint || undefined,
    };
  }

  const rows = workingLines.slice(dataStartIdx).filter(line => line.trim()).map(line => {
    return line.split(delimiter).map(v => {
      const trimmed = v.trim().replace(/^"|"$/g, "");
      const n = parseFloat(trimmed);
      return isNaN(n) ? trimmed : n;
    });
  });

  // Post-parse: detect IBI format from headers
  if (!formatHint) {
    const ibiHint = detectIBIFormat(workingLines, headers);
    if (ibiHint) formatHint = ibiHint;
  }

  return {
    headers,
    rows,
    preview: rows.slice(0, 8),
    rawLines,
    formatHint: formatHint || undefined,
  };
}

// ──────────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────────
function matchesAnyPattern(header: string, patterns: RegExp[]): boolean {
  return patterns.some(p => p.test(header.trim()));
}

function isTimestampLikeColumn(values: (number | string)[]): boolean {
  const nums = values.filter(v => typeof v === "number") as number[];
  if (nums.length < 3) return false;
  let increasing = 0;
  for (let i = 1; i < Math.min(nums.length, 20); i++) {
    if (nums[i] > nums[i - 1]) increasing++;
  }
  return increasing / Math.min(nums.length - 1, 19) > 0.9;
}

function detectPersonPairFromHeader(header: string): { person: 1 | 2 | null; signal: string } {
  // Extended patterns from real OSF datasets:
  // "eeg_alpha_p1", "gaze_x_person2", "hr_participant_1"
  // "mother_hr", "infant_hr", "therapist_eda", "client_eda"
  // "sub01_hbo", "sub02_hbo"
  // "ROI1_sub1", "ROI1_sub2"

  // Suffix-based: _p1, _person1, _s1, _sub1, _participant_a, _member_a
  const p1Suffix = header.match(/[_\-\s](p1|person_?1|s1|subj_?1|sub_?0?1|participant_?[a1]|member_?a|interactor_?1)$/i);
  const p2Suffix = header.match(/[_\-\s](p2|person_?2|s2|subj_?2|sub_?0?2|participant_?[b2]|member_?b|interactor_?2)$/i);

  if (p1Suffix) {
    return { person: 1, signal: header.slice(0, p1Suffix.index!).replace(/[_\-\s]+$/, "") };
  }
  if (p2Suffix) {
    return { person: 2, signal: header.slice(0, p2Suffix.index!).replace(/[_\-\s]+$/, "") };
  }

  // Prefix-based: mother_, therapist_, parent_, caregiver_, adult_, leader_, experimenter_
  const p1Prefix = header.match(/^(mother|mom|mum|parent|therapist|teacher|leader|caregiver|adult|experimenter|father|dad)[_\-\s]/i);
  const p2Prefix = header.match(/^(infant|baby|toddler|child|kid|client|student|follower|peer|patient)[_\-\s]/i);

  if (p1Prefix) {
    return { person: 1, signal: header.slice(p1Prefix[0].length) };
  }
  if (p2Prefix) {
    return { person: 2, signal: header.slice(p2Prefix[0].length) };
  }

  // Prefix-based: sub01_, sub02_ (common in hyperscanning)
  const subPrefix = header.match(/^(sub|subj|part)_?(\d+)[_\-]/i);
  if (subPrefix) {
    const num = parseInt(subPrefix[2]);
    if (num === 1 || num % 2 === 1) return { person: 1, signal: header.slice(subPrefix[0].length) };
    if (num === 2 || num % 2 === 0) return { person: 2, signal: header.slice(subPrefix[0].length) };
  }

  return { person: null, signal: header };
}

// ──────────────────────────────────────────────────────
// Column role detection
// ──────────────────────────────────────────────────────
export function detectColumns(parsed: ParsedCSV): ColumnDetection {
  const { headers, rows } = parsed;
  const detection: ColumnDetection = {
    timestampCols: [],
    person1Cols: [],
    person2Cols: [],
    signalCols: [],
    unknownCols: [],
    confidence: 0,
    suggestions: [],
  };

  // Pass 1: pattern matching on headers
  headers.forEach((header, i) => {
    if (matchesAnyPattern(header, TIMESTAMP_PATTERNS)) {
      detection.timestampCols.push(i);
    } else if (matchesAnyPattern(header, PERSON1_PATTERNS)) {
      detection.person1Cols.push(i);
    } else if (matchesAnyPattern(header, PERSON2_PATTERNS)) {
      detection.person2Cols.push(i);
    } else {
      const { person } = detectPersonPairFromHeader(header);
      if (person === 1) detection.person1Cols.push(i);
      else if (person === 2) detection.person2Cols.push(i);
      else if (matchesAnyPattern(header, SIGNAL_PATTERNS)) {
        detection.signalCols.push(i);
      } else {
        detection.unknownCols.push(i);
      }
    }
  });

  // Pass 2: data-driven timestamp detection
  if (detection.timestampCols.length === 0) {
    for (let i = 0; i < headers.length; i++) {
      const colValues = rows.slice(0, 20).map(r => r[i]);
      if (isTimestampLikeColumn(colValues)) {
        detection.timestampCols.push(i);
        detection.unknownCols = detection.unknownCols.filter(c => c !== i);
        detection.signalCols = detection.signalCols.filter(c => c !== i);
        break;
      }
    }
  }

  // Pass 3: if no person columns, try heuristics
  if (detection.person1Cols.length === 0 && detection.person2Cols.length === 0) {
    const numericCols = headers.map((_, i) => i).filter(i => {
      return !detection.timestampCols.includes(i) &&
        rows.slice(0, 5).every(r => typeof r[i] === "number");
    });

    if (numericCols.length === 2) {
      detection.person1Cols = [numericCols[0]];
      detection.person2Cols = [numericCols[1]];
      detection.suggestions.push("Auto-assigned two numeric columns as Person 1 and Person 2.");
    } else if (numericCols.length >= 2) {
      const paired = findPairedColumns(headers, numericCols);
      if (paired.length > 0) {
        detection.person1Cols = paired.map(p => p[0]);
        detection.person2Cols = paired.map(p => p[1]);
        detection.suggestions.push(`Found ${paired.length} paired signal column(s).`);
      } else {
        // Try correlation-based pairing: if we have an even number, split in half
        if (numericCols.length % 2 === 0 && numericCols.length <= 8) {
          const half = numericCols.length / 2;
          detection.person1Cols = numericCols.slice(0, half);
          detection.person2Cols = numericCols.slice(half);
          detection.suggestions.push(`Split ${numericCols.length} numeric columns into two halves for Person 1 and Person 2. Please verify.`);
        } else {
          detection.suggestions.push("Multiple numeric columns detected. Please manually assign Person 1 and Person 2.");
          detection.signalCols.push(...numericCols);
        }
      }
    }
  }

  // Pass 4: special format hints
  if (parsed.formatHint) {
    const hint = parsed.formatHint;
    if (hint.format === "empatica") {
      detection.suggestions.unshift(`📱 Empatica E4 format detected. Sample rate: ${hint.detectedSampleRate}Hz.`);
    } else if (hint.format === "rmea") {
      detection.suggestions.unshift("🎥 rMEA motion energy format detected (headerless 2-column).");
    } else if (hint.format === "ibi") {
      detection.suggestions.unshift("💓 IBI (Inter-Beat Interval) format: col 1 = time, col 2 = interval.");
    } else if (hint.format === "opensignals") {
      detection.suggestions.unshift("🔬 OpenSignals/BITalino format detected.");
    } else if (hint.format === "biopac") {
      detection.suggestions.unshift("🏥 Biopac AcqKnowledge export detected.");
    } else if (hint.format === "edf") {
      detection.suggestions.unshift("🧠 EDF (European Data Format) file parsed to CSV.");
    }
  }

  // Confidence
  let conf = 0;
  if (detection.timestampCols.length > 0) conf += 0.3;
  if (detection.person1Cols.length > 0) conf += 0.35;
  if (detection.person2Cols.length > 0) conf += 0.35;
  detection.confidence = conf;

  // Suggestions
  if (detection.timestampCols.length === 0) {
    detection.suggestions.push("⚠️ No timestamp column detected. Using row index as time proxy.");
  }
  if (detection.person1Cols.length === 0 || detection.person2Cols.length === 0) {
    detection.suggestions.push("⚠️ Could not detect dyadic pair columns. Assign Person 1 & Person 2 manually.");
  }
  if (detection.person1Cols.length > 1) {
    detection.suggestions.push(
      `ℹ️ Multiple Person 1 signals: ${detection.person1Cols.map(i => headers[i]).join(", ")}.`
    );
  }

  return detection;
}

// ──────────────────────────────────────────────────────
// Paired column finder
// ──────────────────────────────────────────────────────
function findPairedColumns(headers: string[], numericCols: number[]): [number, number][] {
  const pairs: [number, number][] = [];
  const used = new Set<number>();

  for (const i of numericCols) {
    if (used.has(i)) continue;
    const { person: p1, signal: sig1 } = detectPersonPairFromHeader(headers[i]);
    if (p1 !== 1) continue;

    for (const j of numericCols) {
      if (i === j || used.has(j)) continue;
      const { person: p2, signal: sig2 } = detectPersonPairFromHeader(headers[j]);
      if (p2 === 2 && sig1.toLowerCase() === sig2.toLowerCase()) {
        pairs.push([i, j]);
        used.add(i);
        used.add(j);
        break;
      }
    }
  }
  return pairs;
}

// ──────────────────────────────────────────────────────
// Signal groups for multi-signal files
// ──────────────────────────────────────────────────────
export type SignalGroup = {
  signalName: string;
  person1Col: number;
  person2Col: number;
  person1Header: string;
  person2Header: string;
};

export function extractSignalGroups(parsed: ParsedCSV, detection: ColumnDetection): SignalGroup[] {
  const { headers } = parsed;
  const groups: SignalGroup[] = [];

  if (detection.person1Cols.length === detection.person2Cols.length && detection.person1Cols.length > 0) {
    for (let i = 0; i < detection.person1Cols.length; i++) {
      const p1 = detection.person1Cols[i];
      const p2 = detection.person2Cols[i];
      const { signal } = detectPersonPairFromHeader(headers[p1]);
      groups.push({
        signalName: signal || headers[p1],
        person1Col: p1,
        person2Col: p2,
        person1Header: headers[p1],
        person2Header: headers[p2],
      });
    }
  } else if (detection.person1Cols.length === 1 && detection.person2Cols.length === 1) {
    groups.push({
      signalName: "signal",
      person1Col: detection.person1Cols[0],
      person2Col: detection.person2Cols[0],
      person1Header: headers[detection.person1Cols[0]],
      person2Header: headers[detection.person2Cols[0]],
    });
  }

  return groups;
}
