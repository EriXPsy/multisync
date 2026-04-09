import { useState, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
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
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import {
  ArrowRight,
  Database,
  Settings2,
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  Play,
  FileCheck,
  Info,
  Clock,
  Anchor,
} from "lucide-react";
import { toast } from "sonner";
import { DEFAULT_WCC_PARAMS } from "@/lib/synchrony-data";
import type { Json } from "@/integrations/supabase/types";

const STEPS = [
  { id: 1, label: "Select Data", icon: Database },
  { id: 2, label: "Configure", icon: Settings2 },
  { id: 3, label: "Verify Alignment", icon: FileCheck },
  { id: 4, label: "Results", icon: BarChart3 },
];

const MODALITY_COLORS: Record<string, string> = {
  neural: "hsl(262, 60%, 55%)",
  behavioral: "hsl(185, 55%, 40%)",
  bio: "hsl(340, 60%, 55%)",
  psycho: "hsl(35, 80%, 55%)",
};

type StreamData = { t: number; p1: number; p2: number };

function computeWCC(data: StreamData[], windowMs: number, lagMs: number, sampleRateHz: number): number[] {
  const windowSamples = Math.max(1, Math.round((windowMs / 1000) * sampleRateHz));
  const results: number[] = [];
  for (let i = 0; i < data.length - windowSamples; i += Math.max(1, Math.floor(windowSamples / 2))) {
    const w1 = data.slice(i, i + windowSamples).map((d) => d.p1);
    const w2 = data.slice(i, i + windowSamples).map((d) => d.p2);
    const m1 = w1.reduce((a, b) => a + b, 0) / w1.length;
    const m2 = w2.reduce((a, b) => a + b, 0) / w2.length;
    const s1 = Math.sqrt(w1.reduce((a, b) => a + (b - m1) ** 2, 0) / w1.length);
    const s2 = Math.sqrt(w2.reduce((a, b) => a + (b - m2) ** 2, 0) / w2.length);
    if (s1 === 0 || s2 === 0) { results.push(0); continue; }
    let maxCorr = 0;
    const lagSamples = Math.round((lagMs / 1000) * sampleRateHz);
    for (let lag = -lagSamples; lag <= lagSamples; lag++) {
      let sum = 0, count = 0;
      for (let j = 0; j < windowSamples; j++) {
        const j2 = j + lag;
        if (j2 >= 0 && j2 < windowSamples) {
          sum += ((w1[j] - m1) / s1) * ((w2[j2] - m2) / s2);
          count++;
        }
      }
      const corr = count > 0 ? sum / count : 0;
      if (Math.abs(corr) > Math.abs(maxCorr)) maxCorr = corr;
    }
    results.push(maxCorr);
  }
  return results;
}

function zScore(values: number[]): number[] {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  if (std === 0) return values.map(() => 0);
  return values.map((v) => (v - mean) / std);
}

function epochAggregate(values: number[], epochSamples: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < values.length; i += epochSamples) {
    const chunk = values.slice(i, i + epochSamples);
    result.push(chunk.reduce((a, b) => a + b, 0) / chunk.length);
  }
  return result;
}

type StreamOffset = { streamId: string; offsetMs: number; anchorMethod: "none" | "first_nonzero" | "manual" };

const PipelinePage = () => {
  const queryClient = useQueryClient();
  const [step, setStep] = useState(1);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [selectedStreamIds, setSelectedStreamIds] = useState<string[]>([]);
  const [epochMs, setEpochMs] = useState(10000);
  const [normalization, setNormalization] = useState<"zscore" | "minmax">("zscore");
  const [streamOffsets, setStreamOffsets] = useState<Record<string, StreamOffset>>({});
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [alignmentReport, setAlignmentReport] = useState<any>(null);
  const [running, setRunning] = useState(false);

  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: async () => {
      const { data, error } = await supabase.from("datasets").select("*").order("created_at", { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  const { data: streams } = useQuery({
    queryKey: ["data_streams", selectedDatasetId],
    queryFn: async () => {
      if (!selectedDatasetId) return [];
      const { data, error } = await supabase.from("data_streams").select("*").eq("dataset_id", selectedDatasetId);
      if (error) throw error;
      return data;
    },
    enabled: !!selectedDatasetId,
  });

  const toggleStream = (id: string) => {
    setSelectedStreamIds((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
    if (!streamOffsets[id]) {
      setStreamOffsets((prev) => ({
        ...prev,
        [id]: { streamId: id, offsetMs: 0, anchorMethod: "none" },
      }));
    }
  };

  const updateOffset = (id: string, updates: Partial<StreamOffset>) => {
    setStreamOffsets((prev) => ({
      ...prev,
      [id]: { ...prev[id], ...updates },
    }));
  };

  const autoDetectAnchors = () => {
    if (!streams) return;
    const selected = streams.filter((s) => selectedStreamIds.includes(s.id));
    
    // Find earliest timestamp across all streams to use as reference
    let globalMinT = Infinity;
    const streamMinTs: Record<string, number> = {};
    
    for (const stream of selected) {
      const rawData = (stream.data as StreamData[]) || [];
      if (rawData.length === 0) continue;
      const minT = Math.min(...rawData.slice(0, 100).map((d) => d.t));
      streamMinTs[stream.id] = minT;
      if (minT < globalMinT) globalMinT = minT;
    }

    // Set offsets so all streams align to the earliest start
    const newOffsets = { ...streamOffsets };
    for (const stream of selected) {
      const streamMin = streamMinTs[stream.id] ?? 0;
      newOffsets[stream.id] = {
        streamId: stream.id,
        offsetMs: -(streamMin - globalMinT),
        anchorMethod: "first_nonzero",
      };
    }
    setStreamOffsets(newOffsets);
    toast.success("Auto-detected temporal anchors based on earliest timestamps");
  };

  const runAnalysis = async () => {
    if (!selectedDatasetId || selectedStreamIds.length === 0) {
      toast.error("Select data streams first");
      return;
    }

    setRunning(true);
    try {
      const selectedStreams = streams?.filter((s) => selectedStreamIds.includes(s.id)) || [];
      const report: any = { streams: [], alignment: {}, offsets: {} };
      const modalityResults: Record<string, number[]> = {};

      for (const stream of selectedStreams) {
        let rawData = (stream.data as StreamData[]) || [];
        if (rawData.length === 0) continue;

        // Apply temporal offset
        const offset = streamOffsets[stream.id]?.offsetMs || 0;
        if (offset !== 0) {
          rawData = rawData.map((d) => ({ ...d, t: d.t + offset }));
        }

        const sampleRate = stream.sample_rate_hz || 30;
        const windowMs = 5000;
        const lagMs = 2000;

        const wccValues = computeWCC(rawData, windowMs, lagMs, sampleRate);
        const zScored = zScore(wccValues);
        const samplesPerEpoch = Math.max(1, Math.round((epochMs / 1000) * (wccValues.length / (rawData.length / sampleRate))));
        const epoched = epochAggregate(zScored, Math.max(1, samplesPerEpoch));

        modalityResults[stream.modality] = modalityResults[stream.modality] || [];
        if (modalityResults[stream.modality].length === 0) {
          modalityResults[stream.modality] = epoched;
        } else {
          const existing = modalityResults[stream.modality];
          const minLen = Math.min(existing.length, epoched.length);
          modalityResults[stream.modality] = existing.slice(0, minLen).map((v, i) => (v + epoched[i]) / 2);
        }

        const tMin = rawData[0]?.t ?? 0;
        const tMax = rawData[rawData.length - 1]?.t ?? 0;

        report.streams.push({
          name: stream.index_name,
          modality: stream.modality,
          rawSamples: rawData.length,
          wccWindows: wccValues.length,
          epochedPoints: epoched.length,
          sampleRateHz: sampleRate,
          nativeWindowMs: windowMs,
          offsetApplied: offset,
          timeRange: { startMs: tMin, endMs: tMax, durationMs: tMax - tMin },
        });
        report.offsets[stream.id] = offset;
      }

      const maxEpochs = Math.max(...Object.values(modalityResults).map((v) => v.length), 1);
      const chartData = Array.from({ length: maxEpochs }, (_, i) => {
        const point: Record<string, any> = {
          epoch: i,
          time: `${((i * epochMs) / 60000).toFixed(1)}m`,
        };
        for (const [mod, vals] of Object.entries(modalityResults)) {
          point[mod] = vals[i] ?? null;
        }
        return point;
      });

      report.alignment = {
        commonEpochMs: epochMs,
        totalEpochs: maxEpochs,
        modalities: Object.keys(modalityResults),
        normalization,
      };

      setAlignmentReport(report);
      setAnalysisResults(chartData);

      await supabase.from("analysis_runs").insert({
        dataset_id: selectedDatasetId,
        name: `Analysis ${new Date().toLocaleString()}`,
        config: { epochMs, normalization, streamIds: selectedStreamIds, offsets: report.offsets } as unknown as Json,
        status: "complete",
        results: chartData as unknown as Json,
        alignment_report: report as unknown as Json,
      });

      setStep(4);
      toast.success("Analysis complete");
    } catch (err: any) {
      toast.error(err.message || "Analysis failed");
    } finally {
      setRunning(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="p-6 max-w-[1100px] space-y-6">
        <div>
          <h2 className="font-heading text-2xl font-bold">Analysis Pipeline</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Select data → configure methods → verify alignment → view results
          </p>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center gap-2 flex-wrap">
          {STEPS.map((s, i) => (
            <div key={s.id} className="flex items-center gap-2">
              <Button
                variant={step === s.id ? "default" : step > s.id ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setStep(s.id)}
                className="gap-1.5 text-xs"
              >
                {step > s.id ? <CheckCircle2 className="w-3.5 h-3.5" /> : <s.icon className="w-3.5 h-3.5" />}
                {s.label}
              </Button>
              {i < STEPS.length - 1 && <ArrowRight className="w-3 h-3 text-muted-foreground" />}
            </div>
          ))}
        </div>

        {/* Step 1 */}
        {step === 1 && (
          <Card className="glass-panel p-5 space-y-4">
            <h3 className="font-heading text-sm font-semibold">Step 1: Select Dataset & Streams</h3>
            <div className="space-y-2">
              <Label className="text-xs font-heading">Dataset</Label>
              <Select value={selectedDatasetId} onValueChange={(v) => { setSelectedDatasetId(v); setSelectedStreamIds([]); }}>
                <SelectTrigger className="h-9 text-sm"><SelectValue placeholder="Choose a dataset..." /></SelectTrigger>
                <SelectContent>
                  {datasets?.filter((d) => d.status === "complete").map((ds) => (
                    <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {streams && streams.length > 0 && (
              <div className="space-y-2">
                <Label className="text-xs font-heading">Data Streams</Label>
                <div className="space-y-1">
                  {streams.map((s) => (
                    <div key={s.id} className="flex items-center gap-3 p-2 rounded-md bg-muted/30 hover:bg-muted/50">
                      <Checkbox
                        checked={selectedStreamIds.includes(s.id)}
                        onCheckedChange={() => toggleStream(s.id)}
                      />
                      <div className="flex-1">
                        <span className="text-sm font-medium">{s.index_name}</span>
                        <Badge variant="outline" className="ml-2 text-[10px] capitalize">{s.modality}</Badge>
                      </div>
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {s.sample_rate_hz}Hz · {(s.data as any[])?.length || 0} samples
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <Button onClick={() => setStep(2)} disabled={selectedStreamIds.length === 0}>
              Next: Configure <ArrowRight className="w-4 h-4 ml-1" />
            </Button>
          </Card>
        )}

        {/* Step 2 */}
        {step === 2 && (
          <Card className="glass-panel p-5 space-y-4">
            <h3 className="font-heading text-sm font-semibold">Step 2: Configure Analysis</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label className="text-xs font-heading">Common Epoch Resolution</Label>
                <div className="flex items-center gap-3">
                  <Slider
                    value={[epochMs]}
                    onValueChange={([v]) => setEpochMs(v)}
                    min={1000}
                    max={30000}
                    step={1000}
                    className="flex-1"
                  />
                  <Badge variant="secondary" className="min-w-[60px] text-center">{epochMs / 1000}s</Badge>
                </div>
                <p className="text-[10px] text-muted-foreground">
                  All streams epoch-aggregated to this resolution after WCC at native rates.
                </p>
              </div>
              <div className="space-y-2">
                <Label className="text-xs font-heading">Normalization</Label>
                <Select value={normalization} onValueChange={(v) => setNormalization(v as any)}>
                  <SelectTrigger className="h-9 text-sm"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="zscore">Z-Score (recommended)</SelectItem>
                    <SelectItem value="minmax">Min-Max [0,1]</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Separator />

            {/* Temporal Alignment */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Anchor className="w-4 h-4 text-accent" />
                  <h4 className="font-heading text-sm font-semibold">Temporal Alignment</h4>
                </div>
                <Button variant="outline" size="sm" onClick={autoDetectAnchors} className="text-xs gap-1">
                  <Clock className="w-3 h-3" />
                  Auto-Detect Anchors
                </Button>
              </div>
              <p className="text-[10px] text-muted-foreground">
                Set per-stream time offsets to align recordings that started at different absolute times. 
                Like synchronizing instruments in an orchestra — each may have started warming up at different moments, 
                but the conductor's downbeat is the shared anchor.
              </p>

              {selectedStreamIds.map((id) => {
                const stream = streams?.find((s) => s.id === id);
                if (!stream) return null;
                const offset = streamOffsets[id] || { offsetMs: 0, anchorMethod: "none" };
                const rawData = (stream.data as StreamData[]) || [];
                const tMin = rawData[0]?.t ?? 0;
                const tMax = rawData[rawData.length - 1]?.t ?? 0;

                return (
                  <div key={id} className="flex items-center gap-3 p-2 rounded-md bg-muted/20">
                    <div
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: MODALITY_COLORS[stream.modality] || "hsl(var(--accent))" }}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-medium truncate">{stream.index_name}</p>
                      <p className="text-[9px] text-muted-foreground font-mono">
                        {(tMin / 1000).toFixed(1)}s – {(tMax / 1000).toFixed(1)}s ({((tMax - tMin) / 1000).toFixed(1)}s)
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Tooltip>
                        <TooltipTrigger>
                          <Info className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="text-xs max-w-[200px]">
                          Positive offset shifts this stream forward in time; negative shifts it backward. Use to align with a shared event (e.g., task onset, video start).
                        </TooltipContent>
                      </Tooltip>
                      <Input
                        type="number"
                        value={offset.offsetMs}
                        onChange={(e) => updateOffset(id, { offsetMs: parseFloat(e.target.value) || 0 })}
                        className="h-7 w-24 text-[10px]"
                        placeholder="0"
                      />
                      <span className="text-[9px] text-muted-foreground">ms</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <Separator />

            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setStep(1)}>Back</Button>
              <Button onClick={() => { setStep(3); runAnalysis(); }}>
                <Play className="w-4 h-4 mr-1" /> Run Analysis
              </Button>
            </div>
          </Card>
        )}

        {/* Step 3 */}
        {step === 3 && (
          <Card className="glass-panel p-5 space-y-4">
            <h3 className="font-heading text-sm font-semibold">Step 3: Alignment Verification</h3>
            {running ? (
              <div className="text-center py-8">
                <div className="animate-spin w-8 h-8 border-2 border-accent border-t-transparent rounded-full mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">Computing WCC and aligning timescales...</p>
              </div>
            ) : alignmentReport ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {alignmentReport.streams?.map((s: any, i: number) => (
                    <div key={i} className="bg-muted/30 rounded-lg p-3 space-y-1">
                      <p className="text-xs font-heading font-semibold">{s.name}</p>
                      <Badge variant="outline" className="text-[10px] capitalize">{s.modality}</Badge>
                      <div className="text-[10px] text-muted-foreground space-y-0.5 mt-2">
                        <p>Raw: {s.rawSamples} samples @ {s.sampleRateHz}Hz</p>
                        <p>WCC: {s.wccWindows} windows ({s.nativeWindowMs}ms)</p>
                        <p>Epoched: {s.epochedPoints} points @ {epochMs / 1000}s</p>
                        {s.offsetApplied !== 0 && (
                          <p className="text-accent">Offset: {s.offsetApplied > 0 ? "+" : ""}{s.offsetApplied}ms</p>
                        )}
                        {s.timeRange && (
                          <p>Range: {(s.timeRange.startMs / 1000).toFixed(1)}s – {(s.timeRange.endMs / 1000).toFixed(1)}s</p>
                        )}
                      </div>
                      <div className="flex items-center gap-1 mt-1">
                        <CheckCircle2 className="w-3 h-3 text-green-500" />
                        <span className="text-[10px] text-green-600">Aligned</span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="bg-muted/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <FileCheck className="w-4 h-4 text-accent" />
                    <span className="text-xs font-heading font-semibold">Alignment Summary</span>
                  </div>
                  <div className="text-[10px] text-muted-foreground space-y-1">
                    <p>Common epoch: {alignmentReport.alignment?.commonEpochMs}ms</p>
                    <p>Total epochs: {alignmentReport.alignment?.totalEpochs}</p>
                    <p>Modalities aligned: {alignmentReport.alignment?.modalities?.join(", ")}</p>
                    <p>Normalization: {alignmentReport.alignment?.normalization}</p>
                  </div>
                </div>

                <Button onClick={() => setStep(4)}>
                  View Results <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No analysis run yet.</p>
            )}
          </Card>
        )}

        {/* Step 4 */}
        {step === 4 && analysisResults && (
          <div className="space-y-4">
            <Card className="glass-panel p-4">
              <h3 className="font-heading text-sm font-semibold mb-3">Unified Timeline — Real Data</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={analysisResults} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
                  <YAxis
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    label={{
                      value: normalization === "zscore" ? "Z-score" : "Sync [0–1]",
                      angle: -90,
                      position: "insideLeft",
                      style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
                    }}
                  />
                  <ChartTooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "var(--radius)",
                      fontSize: 11,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="hsl(var(--border))" strokeDasharray="2 2" />
                  {Object.keys(MODALITY_COLORS).map((mod) => (
                    analysisResults[0]?.[mod] !== undefined && (
                      <Line
                        key={mod}
                        type="monotone"
                        dataKey={mod}
                        name={`${mod.charAt(0).toUpperCase() + mod.slice(1)} Sync`}
                        stroke={MODALITY_COLORS[mod]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, strokeWidth: 0 }}
                        connectNulls
                      />
                    )
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* Alignment Quality Card */}
            {alignmentReport && (
              <Card className="glass-panel p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Anchor className="w-4 h-4 text-accent" />
                  <h3 className="font-heading text-sm font-semibold">Alignment Quality Report</h3>
                </div>
                <div className="text-xs text-muted-foreground space-y-1">
                  <p>
                    <strong>How alignment works:</strong> Each stream computes synchrony (WCC) at its native temporal resolution, 
                    then the resulting synchrony timeseries is epoch-aggregated to the common resolution ({epochMs / 1000}s). 
                    Like a concert where each instrument plays at its own tempo, but the conductor 
                    (epoch aggregation) ensures all parts align on shared beat boundaries.
                  </p>
                  {alignmentReport.streams?.some((s: any) => s.offsetApplied !== 0) && (
                    <p>
                      <strong>Temporal offsets applied:</strong> Streams were shifted to align their recording start times. 
                      This corrects for equipment startup delays, different trigger signals, or manual recording offsets.
                    </p>
                  )}
                  <p className="text-[10px]">
                    <strong>Tip:</strong> If alignment looks wrong, go back to Step 2 and adjust per-stream offsets manually, 
                    or re-import data with corrected time columns in the Import Portal.
                  </p>
                </div>
              </Card>
            )}

            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setStep(2)}>Adjust Parameters</Button>
              <Button variant="outline" onClick={() => setStep(1)}>New Analysis</Button>
            </div>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
};

export default PipelinePage;
