// Smart CSV parser with automatic column detection for dyadic synchrony data

export type ParsedCSV = {
  headers: string[];
  rows: (number | string)[][];
  preview: (number | string)[][];
  rawLines: string[];
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

// Patterns for auto-detecting column roles
const TIMESTAMP_PATTERNS = [
  /^(time|timestamp|t|ms|seconds?|secs?|epoch|frame|sample_?n?|onset)$/i,
  /time_?(stamp)?/i,
  /^t_?\d*/i,
  /elapsed/i,
  /^frame_?/i,
];

const PERSON1_PATTERNS = [
  /^(p1|person_?1|subj(ect)?_?1|participant_?1|child|speaker_?1|s1|dyad_?1)$/i,
  /person_?1/i,
  /participant_?a/i,
  /^(left|first|mother|parent|therapist|teacher|leader)/i,
  /_p1$/i,
  /_s1$/i,
  /_person1$/i,
];

const PERSON2_PATTERNS = [
  /^(p2|person_?2|subj(ect)?_?2|participant_?2|partner|speaker_?2|s2|dyad_?2)$/i,
  /person_?2/i,
  /participant_?b/i,
  /^(right|second|infant|client|student|follower)/i,
  /_p2$/i,
  /_s2$/i,
  /_person2$/i,
];

const SIGNAL_PATTERNS = [
  /^(eeg|fnirs|hrv|eda|scl|scr|gsr|emg|resp|ecg|ppg|ibi|hr|bpm|gaze|pupil)/i,
  /(alpha|beta|theta|gamma|delta|mu)/i,
  /(hbo|hbr|hbt)/i, // fNIRS
  /(x|y|z)_?(coord|pos|vel|acc)/i,
  /(pitch|yaw|roll)/i,
  /^(au|action_?unit)/i,
  /channel_?\d+/i,
  /^(signal|value|amplitude|power|coherence|sync)/i,
];

function matchesAnyPattern(header: string, patterns: RegExp[]): boolean {
  return patterns.some((p) => p.test(header.trim()));
}

function isTimestampLikeColumn(values: (number | string)[]): boolean {
  const nums = values.filter((v) => typeof v === "number") as number[];
  if (nums.length < 3) return false;
  // Check monotonically increasing
  let increasing = 0;
  for (let i = 1; i < Math.min(nums.length, 20); i++) {
    if (nums[i] > nums[i - 1]) increasing++;
  }
  return increasing / Math.min(nums.length - 1, 19) > 0.9;
}

function detectPersonPairFromHeader(header: string): { person: 1 | 2 | null; signal: string } {
  // Detect patterns like "eeg_alpha_p1", "gaze_x_person2", "hr_participant_1"
  const p1Match = header.match(/[_\-\s](p1|person_?1|s1|subj_?1|participant_?[a1])$/i);
  const p2Match = header.match(/[_\-\s](p2|person_?2|s2|subj_?2|participant_?[b2])$/i);
  
  if (p1Match) {
    return { person: 1, signal: header.slice(0, p1Match.index!).replace(/[_\-\s]+$/, "") };
  }
  if (p2Match) {
    return { person: 2, signal: header.slice(0, p2Match.index!).replace(/[_\-\s]+$/, "") };
  }
  return { person: null, signal: header };
}

export function parseCSV(text: string): ParsedCSV {
  const rawLines = text.trim().split(/\r?\n/);
  const delimiter = rawLines[0].includes("\t") ? "\t" : ",";
  
  const headers = rawLines[0].split(delimiter).map((h) => h.trim().replace(/^"|"$/g, ""));
  
  const rows = rawLines.slice(1).filter((line) => line.trim()).map((line) => {
    return line.split(delimiter).map((v) => {
      const trimmed = v.trim().replace(/^"|"$/g, "");
      const n = parseFloat(trimmed);
      return isNaN(n) ? trimmed : n;
    });
  });
  
  return {
    headers,
    rows,
    preview: rows.slice(0, 8),
    rawLines,
  };
}

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

  // First pass: pattern matching on headers
  headers.forEach((header, i) => {
    if (matchesAnyPattern(header, TIMESTAMP_PATTERNS)) {
      detection.timestampCols.push(i);
    } else if (matchesAnyPattern(header, PERSON1_PATTERNS)) {
      detection.person1Cols.push(i);
    } else if (matchesAnyPattern(header, PERSON2_PATTERNS)) {
      detection.person2Cols.push(i);
    } else {
      // Check for compound names like "eeg_alpha_p1"
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

  // Second pass: if no timestamp found, check data patterns
  if (detection.timestampCols.length === 0) {
    for (let i = 0; i < headers.length; i++) {
      const colValues = rows.slice(0, 20).map((r) => r[i]);
      if (isTimestampLikeColumn(colValues)) {
        detection.timestampCols.push(i);
        // Remove from other lists
        detection.unknownCols = detection.unknownCols.filter((c) => c !== i);
        detection.signalCols = detection.signalCols.filter((c) => c !== i);
        break;
      }
    }
  }

  // Third pass: if no person columns found, try alternating pattern for 2-col files
  if (detection.person1Cols.length === 0 && detection.person2Cols.length === 0) {
    const numericCols = headers.map((_, i) => i).filter((i) => {
      return !detection.timestampCols.includes(i) &&
        rows.slice(0, 5).every((r) => typeof r[i] === "number");
    });
    
    if (numericCols.length === 2) {
      detection.person1Cols = [numericCols[0]];
      detection.person2Cols = [numericCols[1]];
      detection.suggestions.push("Auto-assigned two numeric columns as Person 1 and Person 2.");
    } else if (numericCols.length >= 2) {
      // Check for paired columns (signal_p1, signal_p2 pattern)
      const paired = findPairedColumns(headers, numericCols);
      if (paired.length > 0) {
        detection.person1Cols = paired.map((p) => p[0]);
        detection.person2Cols = paired.map((p) => p[1]);
        detection.suggestions.push(`Found ${paired.length} paired signal column(s) for Person 1 and Person 2.`);
      } else {
        detection.suggestions.push("Multiple numeric columns detected. Please manually assign Person 1 and Person 2 columns.");
        detection.signalCols.push(...numericCols);
      }
    }
  }

  // Compute confidence
  let conf = 0;
  if (detection.timestampCols.length > 0) conf += 0.3;
  if (detection.person1Cols.length > 0) conf += 0.35;
  if (detection.person2Cols.length > 0) conf += 0.35;
  detection.confidence = conf;

  // Generate suggestions
  if (detection.timestampCols.length === 0) {
    detection.suggestions.push("⚠️ No timestamp column detected. Using row index as time proxy.");
  }
  if (detection.person1Cols.length === 0 || detection.person2Cols.length === 0) {
    detection.suggestions.push("⚠️ Could not detect dyadic pair columns. Assign Person 1 & Person 2 manually.");
  }
  if (detection.person1Cols.length > 1) {
    detection.suggestions.push(`ℹ️ Multiple Person 1 signals: ${detection.person1Cols.map((i) => headers[i]).join(", ")}. Multi-signal import supported.`);
  }

  return detection;
}

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

// Extract grouped signal pairs for multi-signal files
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
  
  // If we have equal person1 and person2 cols, pair them
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
