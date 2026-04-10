// EDF (European Data Format) binary file parser for EEG/biosignal data
// Spec: https://www.edfplus.info/specs/edf.html

export type EDFHeader = {
  version: string;
  patientId: string;
  recordingId: string;
  startDate: string;
  startTime: string;
  headerBytes: number;
  reserved: string;
  numDataRecords: number;
  dataRecordDuration: number; // seconds
  numSignals: number;
};

export type EDFSignal = {
  label: string;
  transducerType: string;
  physicalDimension: string;
  physicalMin: number;
  physicalMax: number;
  digitalMin: number;
  digitalMax: number;
  prefiltering: string;
  numSamples: number; // samples per data record
  reserved: string;
};

export type EDFFile = {
  header: EDFHeader;
  signals: EDFSignal[];
  data: number[][]; // signal index → sample values (physical units)
  sampleRates: number[]; // Hz per signal
  durationSeconds: number;
};

function readAscii(view: DataView, offset: number, length: number): string {
  let s = "";
  for (let i = 0; i < length; i++) {
    s += String.fromCharCode(view.getUint8(offset + i));
  }
  return s.trim();
}

export function parseEDF(buffer: ArrayBuffer): EDFFile {
  const view = new DataView(buffer);
  
  // Parse main header (256 bytes)
  const header: EDFHeader = {
    version: readAscii(view, 0, 8),
    patientId: readAscii(view, 8, 80),
    recordingId: readAscii(view, 88, 80),
    startDate: readAscii(view, 168, 8),
    startTime: readAscii(view, 176, 8),
    headerBytes: parseInt(readAscii(view, 184, 8)) || 0,
    reserved: readAscii(view, 192, 44),
    numDataRecords: parseInt(readAscii(view, 236, 8)) || 0,
    dataRecordDuration: parseFloat(readAscii(view, 244, 8)) || 1,
    numSignals: parseInt(readAscii(view, 252, 4)) || 0,
  };

  const ns = header.numSignals;
  let offset = 256;

  // Parse signal headers (ns * 256 bytes total, but stored field-by-field)
  const signals: EDFSignal[] = Array.from({ length: ns }, () => ({
    label: "", transducerType: "", physicalDimension: "",
    physicalMin: 0, physicalMax: 0, digitalMin: 0, digitalMax: 0,
    prefiltering: "", numSamples: 0, reserved: "",
  }));

  const readField = (len: number): string[] => {
    const vals: string[] = [];
    for (let i = 0; i < ns; i++) {
      vals.push(readAscii(view, offset, len));
      offset += len;
    }
    return vals;
  };

  const labels = readField(16);
  const transducers = readField(80);
  const dimensions = readField(8);
  const physMins = readField(8);
  const physMaxs = readField(8);
  const digMins = readField(8);
  const digMaxs = readField(8);
  const prefilters = readField(80);
  const numSamples = readField(8);
  readField(32); // reserved per signal

  for (let i = 0; i < ns; i++) {
    signals[i] = {
      label: labels[i],
      transducerType: transducers[i],
      physicalDimension: dimensions[i],
      physicalMin: parseFloat(physMins[i]) || 0,
      physicalMax: parseFloat(physMaxs[i]) || 0,
      digitalMin: parseInt(digMins[i]) || 0,
      digitalMax: parseInt(digMaxs[i]) || 0,
      prefiltering: prefilters[i],
      numSamples: parseInt(numSamples[i]) || 0,
      reserved: "",
    };
  }

  // Compute sample rates
  const sampleRates = signals.map(s => s.numSamples / header.dataRecordDuration);

  // Parse data records (16-bit signed integers)
  const data: number[][] = signals.map(() => []);
  const dataOffset = header.headerBytes;
  const samplesPerRecord = signals.map(s => s.numSamples);
  const recordSize = samplesPerRecord.reduce((a, b) => a + b, 0) * 2; // 2 bytes per sample

  for (let rec = 0; rec < header.numDataRecords; rec++) {
    let recOffset = dataOffset + rec * recordSize;
    for (let sig = 0; sig < ns; sig++) {
      const s = signals[sig];
      const scale = (s.physicalMax - s.physicalMin) / (s.digitalMax - s.digitalMin);
      const dcOffset = s.physicalMin - s.digitalMin * scale;
      
      for (let samp = 0; samp < s.numSamples; samp++) {
        if (recOffset + 1 < buffer.byteLength) {
          const digitalValue = view.getInt16(recOffset, true); // little-endian
          data[sig].push(digitalValue * scale + dcOffset);
        }
        recOffset += 2;
      }
    }
  }

  return {
    header,
    signals,
    data,
    sampleRates,
    durationSeconds: header.numDataRecords * header.dataRecordDuration,
  };
}

// Convert EDF to CSV-like ParsedCSV format for the import pipeline
import type { ParsedCSV } from "./csv-parser";

export function edfToParsedCSV(edf: EDFFile): ParsedCSV {
  // Filter out annotation channels (EDF+ status/annotation signals)
  const validSignals = edf.signals
    .map((s, i) => ({ signal: s, index: i }))
    .filter(({ signal }) => !signal.label.match(/^(EDF Annotations|Status|Annotation)/i));

  // Use the highest sample rate to build a timestamp column
  const maxRate = Math.max(...validSignals.map(({ index }) => edf.sampleRates[index]));
  const totalSamples = Math.max(...validSignals.map(({ index }) => edf.data[index].length));

  const headers = ["timestamp_s", ...validSignals.map(({ signal }) => signal.label)];
  const rows: (number | string)[][] = [];

  for (let i = 0; i < totalSamples; i++) {
    const t = i / maxRate;
    const row: (number | string)[] = [t];
    for (const { index } of validSignals) {
      // Resample lower-rate signals by repeating (nearest neighbor)
      const ratio = edf.sampleRates[index] / maxRate;
      const srcIdx = Math.min(Math.floor(i * ratio), edf.data[index].length - 1);
      row.push(edf.data[index][srcIdx] ?? 0);
    }
    rows.push(row);
  }

  return {
    headers,
    rows,
    preview: rows.slice(0, 8),
    rawLines: [`[EDF: ${edf.header.patientId} | ${edf.signals.length} channels | ${edf.durationSeconds}s]`],
  };
}
