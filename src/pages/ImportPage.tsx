import { useState, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { supabase } from "@/integrations/supabase/client";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Upload,
  FileSpreadsheet,
  Plus,
  Trash2,
  Check,
  AlertCircle,
  Database,
  Sparkles,
  Info,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";
import { SYNCHRONY_INDICES } from "@/lib/synchrony-data";
import {
  parseCSV,
  detectColumns,
  extractSignalGroups,
  type ParsedCSV,
  type ColumnDetection,
  type SignalGroup,
} from "@/lib/csv-parser";
import { parseEDF, edfToParsedCSV } from "@/lib/edf-parser";

type StreamMapping = {
  file: File;
  parsed: ParsedCSV;
  detection: ColumnDetection;
  signalGroups: SignalGroup[];
  modality: string;
  indexName: string;
  timestampCol: number;
  person1Col: number;
  person2Col: number;
  sampleRateHz: number;
  unit: string;
  timeOffsetMs: number; // temporal anchor offset
  timeUnit: "ms" | "s" | "samples";
};

const ImportPage = () => {
  const queryClient = useQueryClient();
  const [datasetName, setDatasetName] = useState("");
  const [datasetDesc, setDatasetDesc] = useState("");
  const [streams, setStreams] = useState<StreamMapping[]>([]);
  const [uploading, setUploading] = useState(false);

  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("datasets")
        .select("*")
        .order("created_at", { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    for (const file of Array.from(files)) {
      let parsed: ParsedCSV;

      // Handle EDF binary files
      if (file.name.toLowerCase().endsWith(".edf")) {
        try {
          const buffer = await file.arrayBuffer();
          const edf = parseEDF(buffer);
          parsed = edfToParsedCSV(edf);
          parsed.formatHint = {
            format: "edf",
            description: `EDF: ${edf.signals.length} channels, ${edf.durationSeconds}s, patient: ${edf.header.patientId}`,
            detectedSampleRate: Math.max(...edf.sampleRates),
          };
        } catch (err: any) {
          toast.error(`Failed to parse EDF file: ${err.message}`);
          continue;
        }
      } else {
        const text = await file.text();
        parsed = parseCSV(text, file.name);
      }

      const detection = detectColumns(parsed);
      const signalGroups = extractSignalGroups(parsed, detection);

      // Auto-assign columns from detection
      const timestampCol = detection.timestampCols[0] ?? 0;
      const person1Col = detection.person1Cols[0] ?? (timestampCol === 0 ? 1 : 0);
      const person2Col = detection.person2Cols[0] ?? Math.min(person1Col + 1, parsed.headers.length - 1);

      // Try to guess modality from headers
      const headerJoined = parsed.headers.join(" ").toLowerCase();
      let guessedModality = "behavioral";
      if (/eeg|fnirs|nirs|fmri|alpha|theta|beta|gamma|plv|coherence/.test(headerJoined)) guessedModality = "neural";
      else if (/hr|heart|eda|scl|gsr|resp|ibi|bpm|ppg|ecg/.test(headerJoined)) guessedModality = "bio";
      else if (/ios|rapport|flow|likert|rating/.test(headerJoined)) guessedModality = "psycho";

      setStreams((prev) => [
        ...prev,
        {
          file,
          parsed,
          detection,
          signalGroups,
          modality: guessedModality,
          indexName: "",
          timestampCol,
          person1Col,
          person2Col,
          sampleRateHz: parsed.formatHint?.detectedSampleRate ?? (guessedModality === "neural" ? 250 : guessedModality === "bio" ? 4 : 30),
          unit: "a.u.",
          timeOffsetMs: 0,
          timeUnit: parsed.formatHint?.format === "empatica" ? "s" : "ms",
        },
      ]);
    }
    e.target.value = "";
  }, []);

  const updateStream = (idx: number, updates: Partial<StreamMapping>) => {
    setStreams((prev) => prev.map((s, i) => (i === idx ? { ...s, ...updates } : s)));
  };

  const removeStream = (idx: number) => {
    setStreams((prev) => prev.filter((_, i) => i !== idx));
  };

  const redetectColumns = (idx: number) => {
    const stream = streams[idx];
    const detection = detectColumns(stream.parsed);
    const signalGroups = extractSignalGroups(stream.parsed, detection);
    updateStream(idx, {
      detection,
      signalGroups,
      timestampCol: detection.timestampCols[0] ?? stream.timestampCol,
      person1Col: detection.person1Cols[0] ?? stream.person1Col,
      person2Col: detection.person2Cols[0] ?? stream.person2Col,
    });
    toast.success("Re-analyzed column structure");
  };

  const handleImport = async () => {
    if (!datasetName.trim()) {
      toast.error("Please enter a dataset name");
      return;
    }
    if (streams.length === 0) {
      toast.error("Please add at least one data stream");
      return;
    }

    setUploading(true);
    try {
      const modalities = [...new Set(streams.map((s) => s.modality))];
      const { data: dataset, error: dsError } = await supabase
        .from("datasets")
        .insert({
          name: datasetName,
          description: datasetDesc,
          modalities,
          status: "processing",
        })
        .select()
        .single();

      if (dsError) throw dsError;

      for (const stream of streams) {
        const timeMultiplier = stream.timeUnit === "s" ? 1000 : stream.timeUnit === "samples" ? (1000 / stream.sampleRateHz) : 1;

        const timeseriesData = stream.parsed.rows.map((row) => ({
          t: ((Number(row[stream.timestampCol]) || 0) * timeMultiplier) + stream.timeOffsetMs,
          p1: Number(row[stream.person1Col]) || 0,
          p2: Number(row[stream.person2Col]) || 0,
        }));

        const { error } = await supabase.from("data_streams").insert({
          dataset_id: dataset.id,
          modality: stream.modality,
          index_name: stream.indexName || stream.file.name.replace(/\.\w+$/, ""),
          sample_rate_hz: stream.sampleRateHz,
          unit: stream.unit,
          column_mapping: {
            timestamp: stream.parsed.headers[stream.timestampCol],
            person1: stream.parsed.headers[stream.person1Col],
            person2: stream.parsed.headers[stream.person2Col],
          },
          data: timeseriesData,
          metadata: {
            timeOffsetMs: stream.timeOffsetMs,
            timeUnit: stream.timeUnit,
            detectionConfidence: stream.detection.confidence,
          },
        });
        if (error) throw error;
      }

      await supabase.from("datasets").update({ status: "complete" }).eq("id", dataset.id);
      toast.success(`Dataset "${datasetName}" imported with ${streams.length} streams`);
      setDatasetName("");
      setDatasetDesc("");
      setStreams([]);
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    } catch (err: any) {
      toast.error(err.message || "Import failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="p-6 max-w-[1000px] space-y-6">
        <div>
          <h2 className="font-heading text-2xl font-bold">Data Import Portal</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Import CSV/TSV timeseries with smart column detection for multimodal synchrony analysis
          </p>
        </div>

        <Card className="glass-panel p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Plus className="w-4 h-4 text-accent" />
            <h3 className="font-heading text-sm font-semibold">New Dataset</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-xs font-heading">Dataset Name</Label>
              <Input
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                placeholder="e.g., Dyad_001_Session_1"
                className="h-9 text-sm"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-heading">Description</Label>
              <Input
                value={datasetDesc}
                onChange={(e) => setDatasetDesc(e.target.value)}
                placeholder="Optional description"
                className="h-9 text-sm"
              />
            </div>
          </div>

          <Separator />

          <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
            <Upload className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground mb-2">
              Drop files here, or click to browse
            </p>
            <p className="text-[10px] text-muted-foreground mb-3">
              Supports: <strong>CSV/TSV</strong>, <strong>EDF</strong> (EEG), <strong>Empatica E4</strong>, <strong>rMEA</strong>, <strong>OpenSignals</strong>, <strong>Biopac</strong>, <strong>IBI</strong>, and more.
              Auto-detects dyadic columns from headers like <code>mother_hr/infant_hr</code>, <code>sub01_hbo/sub02_hbo</code>, <code>signal_p1/signal_p2</code>.
            </p>
            <label>
              <input
                type="file"
                accept=".csv,.tsv,.txt,.edf"
                multiple
                onChange={handleFileSelect}
                className="hidden"
              />
              <Button variant="outline" size="sm" asChild>
                <span className="cursor-pointer">
                  <FileSpreadsheet className="w-4 h-4 mr-2" />
                  Select Files
                </span>
              </Button>
            </label>
          </div>

          {streams.map((stream, idx) => (
            <Card key={idx} className="p-4 bg-muted/30 space-y-3">
              {/* File Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4 text-accent" />
                  <span className="text-sm font-medium font-heading">{stream.file.name}</span>
                  <Badge variant="secondary" className="text-[10px]">
                    {stream.parsed.headers.length} cols × {stream.parsed.rows.length} rows
                  </Badge>
                </div>
                <div className="flex items-center gap-1">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon" onClick={() => redetectColumns(idx)} className="h-7 w-7">
                        <RefreshCw className="w-3.5 h-3.5" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Re-detect columns</TooltipContent>
                  </Tooltip>
                  <Button variant="ghost" size="icon" onClick={() => removeStream(idx)} className="h-7 w-7">
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>

              {/* Detection Results */}
              {stream.detection && (
                <div className="rounded-md bg-muted/50 p-3 space-y-2">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5 text-accent" />
                    <span className="text-[11px] font-heading font-semibold">Auto-Detection</span>
                    <Badge
                      variant={stream.detection.confidence >= 0.8 ? "default" : stream.detection.confidence >= 0.5 ? "secondary" : "destructive"}
                      className="text-[9px]"
                    >
                      {Math.round(stream.detection.confidence * 100)}% confidence
                    </Badge>
                  </div>
                  {stream.detection.suggestions.length > 0 && (
                    <div className="space-y-0.5">
                      {stream.detection.suggestions.map((s, i) => (
                        <p key={i} className="text-[10px] text-muted-foreground">{s}</p>
                      ))}
                    </div>
                  )}
                  {stream.signalGroups.length > 1 && (
                    <div className="text-[10px] text-accent">
                      ℹ️ Detected {stream.signalGroups.length} paired signal groups: {stream.signalGroups.map((g) => g.signalName).join(", ")}
                    </div>
                  )}
                </div>
              )}

              {/* Modality & Index */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="space-y-1">
                  <Label className="text-[10px]">Modality</Label>
                  <Select value={stream.modality} onValueChange={(v) => updateStream(idx, { modality: v })}>
                    <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="neural">Neural</SelectItem>
                      <SelectItem value="behavioral">Behavioral</SelectItem>
                      <SelectItem value="bio">Biosynchrony</SelectItem>
                      <SelectItem value="psycho">Psycho-synchrony</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">Index Name</Label>
                  <Input
                    value={stream.indexName}
                    onChange={(e) => updateStream(idx, { indexName: e.target.value })}
                    placeholder="e.g., gaze_coordination"
                    className="h-8 text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">Sample Rate (Hz)</Label>
                  <Input
                    type="number"
                    value={stream.sampleRateHz}
                    onChange={(e) => updateStream(idx, { sampleRateHz: parseFloat(e.target.value) || 0 })}
                    className="h-8 text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">Unit</Label>
                  <Input
                    value={stream.unit}
                    onChange={(e) => updateStream(idx, { unit: e.target.value })}
                    className="h-8 text-xs"
                  />
                </div>
              </div>

              {/* Column Mapping */}
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Timestamp Column", key: "timestampCol" as const, detected: stream.detection.timestampCols },
                  { label: "Person 1 Signal", key: "person1Col" as const, detected: stream.detection.person1Cols },
                  { label: "Person 2 Signal", key: "person2Col" as const, detected: stream.detection.person2Cols },
                ].map((col) => (
                  <div key={col.key} className="space-y-1">
                    <div className="flex items-center gap-1">
                      <Label className="text-[10px]">{col.label}</Label>
                      {col.detected.length > 0 && (
                        <Sparkles className="w-2.5 h-2.5 text-accent" />
                      )}
                    </div>
                    <Select
                      value={String(stream[col.key])}
                      onValueChange={(v) => updateStream(idx, { [col.key]: parseInt(v) })}
                    >
                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {stream.parsed.headers.map((h, hi) => (
                          <SelectItem key={hi} value={String(hi)}>
                            {h}
                            {col.detected.includes(hi) ? " ✓" : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>

              {/* Temporal Anchor */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-1">
                    <Label className="text-[10px]">Time Unit</Label>
                    <Tooltip>
                      <TooltipTrigger><Info className="w-2.5 h-2.5 text-muted-foreground" /></TooltipTrigger>
                      <TooltipContent className="max-w-[200px] text-xs">
                        How timestamps are encoded in this file. Milliseconds, seconds, or sample index.
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Select value={stream.timeUnit} onValueChange={(v) => updateStream(idx, { timeUnit: v as any })}>
                    <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ms">Milliseconds</SelectItem>
                      <SelectItem value="s">Seconds</SelectItem>
                      <SelectItem value="samples">Sample Index</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-1">
                    <Label className="text-[10px]">Time Offset (ms)</Label>
                    <Tooltip>
                      <TooltipTrigger><Info className="w-2.5 h-2.5 text-muted-foreground" /></TooltipTrigger>
                      <TooltipContent className="max-w-[250px] text-xs">
                        Shift this stream's timestamps. Use this to align streams that started recording at different times (e.g., EEG started 5s before video → offset = -5000).
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Input
                    type="number"
                    value={stream.timeOffsetMs}
                    onChange={(e) => updateStream(idx, { timeOffsetMs: parseFloat(e.target.value) || 0 })}
                    placeholder="0"
                    className="h-8 text-xs"
                  />
                </div>
              </div>

              {/* Data Preview */}
              <div className="overflow-x-auto">
                <table className="text-[10px] w-full">
                  <thead>
                    <tr>
                      {stream.parsed.headers.map((h, i) => {
                        const isTimestamp = i === stream.timestampCol;
                        const isP1 = i === stream.person1Col;
                        const isP2 = i === stream.person2Col;
                        return (
                          <th
                            key={i}
                            className={`px-2 py-1 text-left font-medium ${
                              isTimestamp ? "text-accent bg-accent/10" :
                              isP1 ? "text-primary bg-primary/10" :
                              isP2 ? "text-secondary-foreground bg-secondary/30" :
                              "text-muted-foreground"
                            }`}
                          >
                            {h}
                            {isTimestamp && " ⏱"}
                            {isP1 && " 👤₁"}
                            {isP2 && " 👤₂"}
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {stream.parsed.preview.map((row, ri) => (
                      <tr key={ri} className="border-t border-border/30">
                        {row.map((val, ci) => (
                          <td key={ci} className="px-2 py-1 font-mono">
                            {typeof val === "number" ? val.toFixed(4) : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ))}

          {streams.length > 0 && (
            <Button onClick={handleImport} disabled={uploading} className="w-full">
              {uploading ? (
                "Importing..."
              ) : (
                <>
                  <Database className="w-4 h-4 mr-2" />
                  Import {streams.length} Stream{streams.length > 1 ? "s" : ""} to Database
                </>
              )}
            </Button>
          )}
        </Card>

        <Card className="glass-panel p-5 space-y-3">
          <h3 className="font-heading text-sm font-semibold">Existing Datasets</h3>
          {!datasets || datasets.length === 0 ? (
            <p className="text-xs text-muted-foreground">No datasets imported yet.</p>
          ) : (
            <div className="space-y-2">
              {datasets.map((ds) => (
                <div key={ds.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                  <div>
                    <p className="text-sm font-medium font-heading">{ds.name}</p>
                    <p className="text-[10px] text-muted-foreground">{ds.description}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {ds.modalities?.map((m) => (
                      <Badge key={m} variant="outline" className="text-[10px] capitalize">{m}</Badge>
                    ))}
                    <Badge
                      variant={ds.status === "complete" ? "default" : "secondary"}
                      className="text-[10px]"
                    >
                      {ds.status === "complete" && <Check className="w-3 h-3 mr-1" />}
                      {ds.status === "error" && <AlertCircle className="w-3 h-3 mr-1" />}
                      {ds.status}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </TooltipProvider>
  );
};

export default ImportPage;
