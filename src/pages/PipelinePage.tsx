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
  TrendingUp,
  Zap,
  Target,
  Music,
} from "lucide-react";
import { toast } from "sonner";
import { DEFAULT_WCC_PARAMS } from "@/lib/synchrony-data";
import { runCascadeAnalysis, type CascadeReport } from "@/lib/cascade-analysis";
import { computeWCCDirectional, normalize, epochAggregate, epochAggregateDirectional, type StreamData, type NormalizationMethod, type EpochDirectionalResult } from "@/lib/wcc-compute";
import { runSurrogateTestBatch, type SurrogateResult } from "@/lib/surrogate-testing";
import { runDynamicFeatureExtraction, type DynamicFeatureReport } from "@/lib/dynamic-features";
import { runPredictionWindowAnalysis, type PredictionResult } from "@/lib/prediction-window";
import type { Json } from "@/integrations/supabase/types";
import ScoreViewPanel, { type ContextSegment } from "@/components/ScoreViewPanel";
import CascadeMapPanel from "@/components/CascadeMapPanel";

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

// StreamData type and computation functions imported from @/lib/wcc-compute

type StreamOffset = { streamId: string; offsetMs: number; anchorMethod: "none" | "first_nonzero" | "manual" };

const PipelinePage = () => {
  const queryClient = useQueryClient();
  const [step, setStep] = useState(1);
  const [selectedStreamIds, setSelectedStreamIds] = useState<string[]>([]);
  const [epochMs, setEpochMs] = useState(10000);
  const [normalization, setNormalization] = useState<NormalizationMethod>("zscore");
  const [baselineMs, setBaselineMs] = useState(60000); // 60s default baseline
  const [streamOffsets, setStreamOffsets] = useState<Record<string, StreamOffset>>({});
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [alignmentReport, setAlignmentReport] = useState<any>(null);
  const [dynamicReport, setDynamicReport] = useState<DynamicFeatureReport | null>(null);
  const [cascadeReport, setCascadeReport] = useState<CascadeReport | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [contextSegments, setContextSegments] = useState<ContextSegment[]>([]);
  const [running, setRunning] = useState(false);

  // Fetch ALL datasets
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: async () => {
      const { data, error } = await supabase.from("datasets").select("*").order("created_at", { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  // Fetch ALL streams across ALL complete datasets
  const { data: allStreams } = useQuery({
    queryKey: ["all_data_streams"],
    queryFn: async () => {
      const { data, error } = await supabase.from("data_streams").select("*");
      if (error) throw error;
      return data;
    },
  });

  // Group streams by dataset for display
  const streamsByDataset = useMemo(() => {
    if (!allStreams || !datasets) return [];
    const completeIds = new Set(datasets.filter((d) => d.status === "complete").map((d) => d.id));
    const groups: { dataset: typeof datasets[0]; streams: typeof allStreams }[] = [];
    for (const ds of datasets) {
      if (!completeIds.has(ds.id)) continue;
      const dsStreams = allStreams.filter((s) => s.dataset_id === ds.id);
      if (dsStreams.length > 0) groups.push({ dataset: ds, streams: dsStreams });
    }
    return groups;
  }, [allStreams, datasets]);

  // Detect cross-dataset selection
  const selectedDatasetIds = useMemo(() => {
    if (!allStreams) return new Set<string>();
    const ids = new Set<string>();
    for (const id of selectedStreamIds) {
      const stream = allStreams.find((s) => s.id === id);
      if (stream) ids.add(stream.dataset_id);
    }
    return ids;
  }, [selectedStreamIds, allStreams]);

  const isCrossDataset = selectedDatasetIds.size > 1;

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
    if (!allStreams) return;
    const selected = allStreams.filter((s) => selectedStreamIds.includes(s.id));

    let globalMinT = Infinity;
    const streamMinTs: Record<string, number> = {};

    for (const stream of selected) {
      const rawData = (stream.data as StreamData[]) || [];
      if (rawData.length === 0) continue;
      const minT = Math.min(...rawData.slice(0, 100).map((d) => d.t));
      streamMinTs[stream.id] = minT;
      if (minT < globalMinT) globalMinT = minT;
    }

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
    if (selectedStreamIds.length === 0) {
      toast.error("Select data streams first");
      return;
    }

    setRunning(true);
    try {
      const selectedStreams = allStreams?.filter((s) => selectedStreamIds.includes(s.id)) || [];
      const report: any = { streams: [], alignment: {}, offsets: {}, crossDataset: isCrossDataset };
      const modalityResults: Record<string, number[]> = {};
      const modalityDirectional: Record<string, EpochDirectionalResult[]> = {};

      for (const stream of selectedStreams) {
        let rawData = (stream.data as StreamData[]) || [];
        if (rawData.length === 0) continue;

        const offset = streamOffsets[stream.id]?.offsetMs || 0;
        if (offset !== 0) {
          rawData = rawData.map((d) => ({ ...d, t: d.t + offset }));
        }

        const sampleRate = stream.sample_rate_hz || 30;
        const windowMs = 5000;
        const lagMs = 2000;

        // Directional WCC
        const wccWindows = computeWCCDirectional(rawData, windowMs, lagMs, sampleRate, true);
        const wccValues = wccWindows.map(w => w.peakAbsCorrelation);
        const wccWindowsPerSec = wccValues.length / (rawData.length / sampleRate);
        const normalized = normalize(wccValues, { method: normalization, baselineEndMs: baselineMs }, wccWindowsPerSec);
        const samplesPerEpoch = Math.max(1, Math.round((epochMs / 1000) * wccWindowsPerSec));
        const epoched = epochAggregate(normalized, Math.max(1, samplesPerEpoch));

        // Directional epoch aggregation
        const epochDir = epochAggregateDirectional(wccWindows, Math.max(1, samplesPerEpoch));

        modalityResults[stream.modality] = modalityResults[stream.modality] || [];
        if (modalityResults[stream.modality].length === 0) {
          modalityResults[stream.modality] = epoched;
          modalityDirectional[stream.modality] = epochDir;
        } else {
          const existing = modalityResults[stream.modality];
          const minLen = Math.min(existing.length, epoched.length);
          modalityResults[stream.modality] = existing.slice(0, minLen).map((v, i) => (v + epoched[i]) / 2);
        }

        const tMin = rawData[0]?.t ?? 0;
        const tMax = rawData[rawData.length - 1]?.t ?? 0;
        const dsName = datasets?.find((d) => d.id === stream.dataset_id)?.name || "Unknown";

        report.streams.push({
          name: stream.index_name,
          modality: stream.modality,
          datasetName: dsName,
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
          // Add directional info
          const dir = modalityDirectional[mod]?.[i];
          if (dir) {
            point[`${mod}_direction`] = dir.direction;
            point[`${mod}_signed`] = dir.meanSigned;
            point[`${mod}_lagMs`] = dir.meanLagMs;
            point[`${mod}_fracNeg`] = dir.fractionNegative;
          }
        }
        return point;
      });

      // Run cascade analysis
      const cascadeRep = runCascadeAnalysis(modalityResults, epochMs);
      setCascadeReport(cascadeRep);

      // Run dynamic feature extraction
      const dynReport = runDynamicFeatureExtraction(modalityResults, epochMs);
      setDynamicReport(dynReport);

      // Run prediction window analysis
      const predResult = runPredictionWindowAnalysis(modalityResults, epochMs, {
        deltaT: Math.max(1, Math.round(20000 / epochMs)), // predict 20s ahead
        thresholdSigma: 0.5,
        useChangeRate: true,
      });
      setPredictionResult(predResult);

      // Run surrogate testing (reduced count for browser performance)
      const surrogateStreams = selectedStreams
        .filter(s => {
          const rawData = (s.data as StreamData[]) || [];
          return rawData.length > 0;
        })
        .map(s => {
          let rawData = (s.data as StreamData[]) || [];
          const offset = streamOffsets[s.id]?.offsetMs || 0;
          if (offset !== 0) rawData = rawData.map(d => ({ ...d, t: d.t + offset }));
          return {
            data: rawData,
            name: s.index_name,
            modality: s.modality,
            sampleRateHz: s.sample_rate_hz || 30,
          };
        });
      const surrogateResults = runSurrogateTestBatch(surrogateStreams, 5000, 2000, 100);

      report.alignment = {
        commonEpochMs: epochMs,
        totalEpochs: maxEpochs,
        modalities: Object.keys(modalityResults),
        normalization,
      };
      report.cascade = cascadeRep;
      report.surrogates = surrogateResults;
      report.dynamic = dynReport;
      report.prediction = predResult;

      setAlignmentReport(report);
      setAnalysisResults(chartData);

      // Save — pick the first dataset_id for the FK
      const primaryDatasetId = selectedStreams[0]?.dataset_id;
      if (primaryDatasetId) {
        await supabase.from("analysis_runs").insert({
          dataset_id: primaryDatasetId,
          name: `Analysis ${new Date().toLocaleString()}`,
          config: { epochMs, normalization, streamIds: selectedStreamIds, offsets: report.offsets, crossDataset: isCrossDataset } as unknown as Json,
          status: "complete",
          results: chartData as unknown as Json,
          alignment_report: report as unknown as Json,
        });
        queryClient.invalidateQueries({ queryKey: ["analysis_runs"] });
      }

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

        {/* Step 1: Cross-dataset stream selection */}
        {step === 1 && (
          <Card className="glass-panel p-5 space-y-4">
            <h3 className="font-heading text-sm font-semibold">Step 1: Select Streams (across all datasets)</h3>
            <p className="text-[11px] text-muted-foreground">
              You can pick streams from different datasets to combine modalities. Streams from different datasets will be marked.
            </p>

            {/* Cross-dataset warning */}
            {isCrossDataset && (
              <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
                <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                <div className="text-xs">
                  <p className="font-semibold text-amber-600">Cross-dataset selection detected</p>
                  <p className="text-muted-foreground mt-0.5">
                    You are combining streams from <strong>{selectedDatasetIds.size} different datasets</strong>. 
                    Please verify that these recordings come from the same session/dyad and that their timelines can be meaningfully aligned. 
                    Use the temporal alignment controls in Step 2 to set proper offsets.
                  </p>
                </div>
              </div>
            )}

            {streamsByDataset.map(({ dataset, streams }) => (
              <div key={dataset.id} className="space-y-2">
                <div className="flex items-center gap-2">
                  <Database className="w-3.5 h-3.5 text-muted-foreground" />
                  <span className="text-xs font-heading font-semibold">{dataset.name}</span>
                  <Badge variant="outline" className="text-[9px]">
                    {dataset.modalities?.join(", ")}
                  </Badge>
                </div>
                <div className="space-y-1 ml-5">
                  {streams.map((s) => (
                    <div key={s.id} className="flex items-center gap-3 p-2 rounded-md bg-muted/30 hover:bg-muted/50">
                      <Checkbox
                        checked={selectedStreamIds.includes(s.id)}
                        onCheckedChange={() => toggleStream(s.id)}
                      />
                      <div
                        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ backgroundColor: MODALITY_COLORS[s.modality] || "hsl(var(--accent))" }}
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
            ))}

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
                {epochMs >= 10000 && (
                  <div className="flex items-start gap-1.5 p-2 rounded-md bg-amber-500/10 border border-amber-500/20 mt-1">
                    <AlertTriangle className="w-3 h-3 text-amber-500 mt-0.5 flex-shrink-0" />
                    <p className="text-[9px] text-amber-600">
                      Large epochs ({epochMs / 1000}s) smooth away fast temporal dynamics. 
                      Neural cascades happening within seconds will be invisible. 
                      Cascade detection is limited to the epoch timescale — consider 1-5s for finer resolution (more epochs = more statistical power for Granger tests).
                    </p>
                  </div>
                )}
                {epochMs <= 2000 && (
                  <div className="flex items-start gap-1.5 p-2 rounded-md bg-blue-500/10 border border-blue-500/20 mt-1">
                    <Info className="w-3 h-3 text-blue-500 mt-0.5 flex-shrink-0" />
                    <p className="text-[9px] text-blue-600">
                      Fine epochs ({epochMs / 1000}s) preserve temporal detail but may be noisy for slow modalities (bio, psycho). 
                      Consider whether your slowest stream has meaningful variation at this timescale.
                    </p>
                  </div>
                )}
              </div>
              <div className="space-y-2">
                <Label className="text-xs font-heading">Normalization</Label>
                <Select value={normalization} onValueChange={(v) => setNormalization(v as NormalizationMethod)}>
                  <SelectTrigger className="h-9 text-sm"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="zscore">Z-Score — session-global (default)</SelectItem>
                    <SelectItem value="zscore_baseline">Z-Score — baseline period</SelectItem>
                    <SelectItem value="minmax">Min-Max [0,1]</SelectItem>
                  </SelectContent>
                </Select>
                {normalization === "zscore_baseline" && (
                  <div className="space-y-1 mt-2 p-2 rounded-md bg-muted/30">
                    <Label className="text-[10px] font-heading">Baseline period (seconds)</Label>
                    <div className="flex items-center gap-3">
                      <Slider
                        value={[baselineMs / 1000]}
                        onValueChange={([v]) => setBaselineMs(v * 1000)}
                        min={10}
                        max={300}
                        step={10}
                        className="flex-1"
                      />
                      <Badge variant="secondary" className="min-w-[50px] text-center">{baselineMs / 1000}s</Badge>
                    </div>
                    <p className="text-[9px] text-muted-foreground">
                      Z-scores computed using mean &amp; SD from the first {baselineMs / 1000}s only. 
                      Values outside this baseline can exceed ±1, reflecting genuine synchrony above resting levels.
                    </p>
                  </div>
                )}
                <p className="text-[9px] text-muted-foreground">
                  {normalization === "zscore" && "Standardizes using session-wide mean/SD. Simple but may dilute strong synchrony periods."}
                  {normalization === "zscore_baseline" && "Standardizes against a resting baseline — preserves genuine high-synchrony signal."}
                  {normalization === "minmax" && "Rescales to [0,1]. Sensitive to outliers; not recommended for cascade detection."}
                </p>
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
              </p>

              {isCrossDataset && (
                <div className="flex items-start gap-2 p-2 rounded-lg bg-amber-500/10 border border-amber-500/30">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-500 mt-0.5 flex-shrink-0" />
                  <p className="text-[10px] text-amber-600">
                    Cross-dataset streams selected — temporal alignment is especially important. Use "Auto-Detect Anchors" or set manual offsets.
                  </p>
                </div>
              )}

              {selectedStreamIds.map((id) => {
                const stream = allStreams?.find((s) => s.id === id);
                if (!stream) return null;
                const offset = streamOffsets[id] || { offsetMs: 0, anchorMethod: "none" };
                const rawData = (stream.data as StreamData[]) || [];
                const tMin = rawData[0]?.t ?? 0;
                const tMax = rawData[rawData.length - 1]?.t ?? 0;
                const dsName = datasets?.find((d) => d.id === stream.dataset_id)?.name || "";

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
                        {dsName && <span className="ml-1 text-accent">· {dsName.slice(0, 30)}</span>}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Tooltip>
                        <TooltipTrigger>
                          <Info className="w-3 h-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent className="text-xs max-w-[200px]">
                          Positive offset shifts this stream forward in time; negative shifts it backward.
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
                      {s.datasetName && (
                        <p className="text-[9px] text-accent truncate">{s.datasetName}</p>
                      )}
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

                {alignmentReport.crossDataset && (
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
                    <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                    <div className="text-xs">
                      <p className="font-semibold text-amber-600">Cross-dataset analysis</p>
                      <p className="text-muted-foreground">
                        Streams from different datasets were combined. Results are valid only if they originate from the same recording session. 
                        Check that time ranges overlap meaningfully.
                      </p>
                    </div>
                  </div>
                )}

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
            {/* Score View — Context Annotation Track */}
            <ScoreViewPanel
              chartData={analysisResults}
              epochMs={epochMs}
              segments={contextSegments}
              onSegmentsChange={setContextSegments}
            />

            {/* Cascade Map — Modality synchrony onset chain */}
            {cascadeReport && (
              <CascadeMapPanel
                onsets={cascadeReport.onsets}
                cascadeOrder={cascadeReport.cascadeOrder}
                leadLagMatrix={cascadeReport.leadLagMatrix}
                grangerResults={cascadeReport.grangerResults}
                sensitivityAnalysis={cascadeReport.sensitivityAnalysis}
                cascadeStability={cascadeReport.cascadeStability}
                dynamicReport={dynamicReport}
              />
            )}

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
                  </p>
                  {alignmentReport.crossDataset && (
                    <p className="text-amber-600">
                      <strong>⚠ Cross-dataset:</strong> Streams were pulled from different datasets. Verify they represent the same experimental session.
                    </p>
                  )}
                  {alignmentReport.streams?.some((s: any) => s.offsetApplied !== 0) && (
                    <p>
                      <strong>Temporal offsets applied:</strong> Streams were shifted to align their recording start times.
                    </p>
                  )}
                </div>
              </Card>
            )}

            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setStep(2)}>Adjust Parameters</Button>
              <Button variant="outline" onClick={() => setStep(1)}>New Analysis</Button>
            </div>

            {/* Dynamic Features Panel */}
            {dynamicReport && (
              <Card className="glass-panel p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-accent" />
                  <h3 className="font-heading text-sm font-semibold">Dynamic Feature Extraction</h3>
                  <Badge variant="outline" className="text-[10px]">Process, not just level</Badge>
                </div>

                <p className="text-xs text-muted-foreground">
                  {dynamicReport.summary}
                </p>

                {/* Per-modality feature table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px]">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-1 pr-3 font-heading font-semibold">Modality</th>
                        <th className="text-right py-1 px-2 font-heading">Onset</th>
                        <th className="text-right py-1 px-2 font-heading">Build-up</th>
                        <th className="text-right py-1 px-2 font-heading">Peak</th>
                        <th className="text-right py-1 px-2 font-heading">Maintain</th>
                        <th className="text-right py-1 px-2 font-heading">Breakdowns</th>
                        <th className="text-right py-1 px-2 font-heading">Recovery</th>
                        <th className="text-right py-1 px-2 font-heading">Entropy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dynamicReport.perModality.map((f) => (
                        <tr key={f.modality} className="border-b border-border/50">
                          <td className="py-1 pr-3 font-medium capitalize">
                            <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: MODALITY_COLORS[f.modality] }} />
                            {f.modality}
                          </td>
                          <td className="text-right py-1 px-2">{f.onsetLatencySec !== null ? `${f.onsetLatencySec}s` : "—"}</td>
                          <td className="text-right py-1 px-2">{f.buildUpRate !== null ? f.buildUpRate.toFixed(3) : "—"}</td>
                          <td className="text-right py-1 px-2">{f.peakValue.toFixed(3)}</td>
                          <td className="text-right py-1 px-2">{f.maintenanceDurationSec !== null ? `${f.maintenanceDurationSec}s` : "—"}</td>
                          <td className="text-right py-1 px-2">
                            <span className={f.breakdownCount >= 3 ? "text-red-500 font-semibold" : ""}>{f.breakdownCount}</span>
                          </td>
                          <td className="text-right py-1 px-2">{f.avgRecoveryEpochs !== null ? `${f.avgRecoveryEpochs.toFixed(1)}e` : "—"}</td>
                          <td className="text-right py-1 px-2">
                            <span className={f.entrainmentEntropy < 2 ? "text-green-600" : f.entrainmentEntropy > 3 ? "text-amber-600" : ""}>
                              {f.entrainmentEntropy.toFixed(2)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Cross-modality features */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <div className="bg-muted/30 rounded-lg p-2 text-center">
                    <p className="text-[9px] text-muted-foreground">Cascade Leader</p>
                    <p className="text-sm font-heading font-semibold capitalize">{dynamicReport.crossModality.leaderModality || "—"}</p>
                  </div>
                  <div className="bg-muted/30 rounded-lg p-2 text-center">
                    <p className="text-[9px] text-muted-foreground">Cascade Delay</p>
                    <p className="text-sm font-heading font-semibold">{dynamicReport.crossModality.cascadeOnsetDelaySec !== null ? `${dynamicReport.crossModality.cascadeOnsetDelaySec}s` : "—"}</p>
                  </div>
                  <div className="bg-muted/30 rounded-lg p-2 text-center">
                    <p className="text-[9px] text-muted-foreground">Multi-sync Epochs</p>
                    <p className="text-sm font-heading font-semibold">{dynamicReport.crossModality.multimodalCoherenceEpochs}</p>
                  </div>
                  <div className="bg-muted/30 rounded-lg p-2 text-center">
                    <p className="text-[9px] text-muted-foreground">Full-Chord %</p>
                    <p className="text-sm font-heading font-semibold">{(dynamicReport.crossModality.fullChordProportion * 100).toFixed(1)}%</p>
                  </div>
                </div>

                <p className="text-[9px] text-muted-foreground italic">
                  Build-up: z-score change per epoch (higher = faster emergence). Entropy: signal regularity (lower = more entrained). Full-Chord: epochs where ALL modalities are synchronized.
                </p>
              </Card>
            )}

            {/* Prediction Window Panel */}
            {predictionResult && (
              <Card className="glass-panel p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-500" />
                  <h3 className="font-heading text-sm font-semibold">Prediction Window Analysis</h3>
                  <Badge variant="outline" className="text-[10px]">Prelude Signal Detection</Badge>
                </div>

                <p className="text-xs text-muted-foreground">
                  Can current synchrony patterns predict multimodal synchrony {predictionResult.deltaTSec}s in the future?
                </p>

                {predictionResult.warnings.map((w, i) => (
                  <div key={i} className="flex items-start gap-1.5 p-2 rounded-md bg-amber-500/10 border border-amber-500/20">
                    <AlertTriangle className="w-3 h-3 text-amber-500 mt-0.5 flex-shrink-0" />
                    <p className="text-[9px] text-amber-600">{w}</p>
                  </div>
                ))}

                <p className="text-xs">{predictionResult.summary}</p>

                {/* Performance metrics */}
                <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                  {[
                    { label: "Accuracy", value: (predictionResult.accuracy * 100).toFixed(1), suffix: "%" },
                    { label: "F1 Score", value: (predictionResult.f1 * 100).toFixed(1), suffix: "%" },
                    { label: "AUC", value: predictionResult.auc.toFixed(3), suffix: "", highlight: predictionResult.auc > 0.7 },
                    { label: "Sensitivity", value: (predictionResult.sensitivity * 100).toFixed(1), suffix: "%" },
                    { label: "Specificity", value: (predictionResult.specificity * 100).toFixed(1), suffix: "%" },
                    { label: "Samples", value: String(predictionResult.totalSamples), suffix: "" },
                  ].map((m) => (
                    <div key={m.label} className={`bg-muted/30 rounded-lg p-2 text-center ${"highlight" in m && m.highlight ? "ring-1 ring-green-500/50" : ""}`}>
                      <p className="text-[9px] text-muted-foreground">{m.label}</p>
                      <p className="text-sm font-heading font-semibold">{m.value}{m.suffix}</p>
                    </div>
                  ))}
                </div>

                {/* Feature importance */}
                {predictionResult.featureImportance.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-heading font-semibold flex items-center gap-1">
                      <Target className="w-3 h-3" /> Top Predictors (Feature Importance)
                    </p>
                    {predictionResult.featureImportance.slice(0, 6).map((f, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <span className="text-[10px] text-muted-foreground w-20 truncate">{f.description}</span>
                        <div className="flex-1 h-3 bg-muted/50 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${f.weight > 0 ? "bg-green-500" : "bg-red-500"}`}
                            style={{
                              width: `${Math.min(100, (Math.abs(f.weight) / Math.max(...predictionResult.featureImportance.map(x => Math.abs(x.weight)))) * 100)}%`,
                            }}
                          />
                        </div>
                        <span className="text-[10px] font-mono w-12 text-right">
                          {f.weight > 0 ? "+" : ""}{f.weight.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Confusion matrix */}
                <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
                  <span>Confusion: TP={predictionResult.tp} FP={predictionResult.fp} TN={predictionResult.tn} FN={predictionResult.fn}</span>
                  <span>Model: {predictionResult.model}</span>
                  <span>Horizon: {predictionResult.deltaTSec}s ahead</span>
                </div>

                <p className="text-[9px] text-muted-foreground italic">
                  This is a proof-of-concept logistic regression with leave-one-out CV. For publication, export features and use scikit-learn with proper nested cross-validation.
                </p>
              </Card>
            )}
          </div>
        )}
      </div>
    </TooltipProvider>
  );
};

export default PipelinePage;
