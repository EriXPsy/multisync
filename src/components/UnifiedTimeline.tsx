import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { supabase } from "@/integrations/supabase/client";
import { useQuery } from "@tanstack/react-query";
import { Brain, Eye, Heart, Users, Info, AlertTriangle, BarChart3, ArrowRight, TrendingUp, Zap, GitBranch } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import type { CascadeReport, OnsetResult, LeadLagResult, GrangerResult, SensitivityResult } from "@/lib/cascade-analysis";

const MODALITY_ICONS: Record<string, any> = {
  neural: Brain,
  behavioral: Eye,
  bio: Heart,
  psycho: Users,
};

const MODALITY_COLORS: Record<string, string> = {
  neural: "hsl(262, 60%, 55%)",
  behavioral: "hsl(185, 55%, 40%)",
  bio: "hsl(340, 60%, 55%)",
  psycho: "hsl(35, 80%, 55%)",
};

const MODALITY_LABELS: Record<string, string> = {
  neural: "Neural",
  behavioral: "Behavioral",
  bio: "Bio",
  psycho: "Psycho",
};

export function UnifiedTimeline() {
  const navigate = useNavigate();

  const { data: analysisRuns, isLoading } = useQuery({
    queryKey: ["analysis_runs"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("analysis_runs")
        .select("*")
        .eq("status", "complete")
        .order("created_at", { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  const [selectedRunId, setSelectedRunId] = useState<string>("");

  const selectedRun = useMemo(() => {
    if (!analysisRuns || analysisRuns.length === 0) return null;
    if (selectedRunId) return analysisRuns.find((r) => r.id === selectedRunId) || null;
    return analysisRuns[0];
  }, [analysisRuns, selectedRunId]);

  const chartData = useMemo(() => {
    if (!selectedRun?.results) return [];
    return selectedRun.results as any[];
  }, [selectedRun]);

  const report = useMemo(() => {
    if (!selectedRun?.alignment_report) return null;
    return selectedRun.alignment_report as any;
  }, [selectedRun]);

  const cascade: CascadeReport | null = useMemo(() => {
    return report?.cascade || null;
  }, [report]);

  const modalities = useMemo(() => {
    if (!chartData || chartData.length === 0) return [];
    return Object.keys(chartData[0]).filter(
      (k) => !["epoch", "time", "timeMs"].includes(k) && !k.endsWith("_conf")
    );
  }, [chartData]);

  const config = useMemo(() => {
    if (!selectedRun?.config) return null;
    return selectedRun.config as any;
  }, [selectedRun]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <h2 className="font-heading text-2xl font-bold text-foreground">Unified Multimodal Timeline</h2>
        <Card className="glass-panel p-8 text-center">
          <div className="animate-spin w-8 h-8 border-2 border-accent border-t-transparent rounded-full mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">Loading analysis results...</p>
        </Card>
      </div>
    );
  }

  if (!analysisRuns || analysisRuns.length === 0) {
    return (
      <div className="space-y-6">
        <h2 className="font-heading text-2xl font-bold text-foreground">Unified Multimodal Timeline</h2>
        <Card className="glass-panel p-8 text-center space-y-4">
          <AlertTriangle className="w-10 h-10 text-muted-foreground mx-auto" />
          <div>
            <p className="text-sm font-medium">No analysis results yet</p>
            <p className="text-xs text-muted-foreground mt-1">
              Run an analysis in the Pipeline to see real synchronized results here.
            </p>
          </div>
          <Button onClick={() => navigate("/pipeline")} className="gap-1.5">
            <BarChart3 className="w-4 h-4" /> Go to Pipeline
          </Button>
        </Card>
      </div>
    );
  }

  const epochMs = config?.epochMs || 10000;
  const normalization = config?.normalization || "zscore";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="font-heading text-2xl font-bold text-foreground">
          Unified Multimodal Timeline
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Epoch-aggregated synchrony with cascade detection and lead-lag analysis.
        </p>
      </div>

      {/* Run Selector */}
      <Card className="glass-panel p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label className="text-xs font-heading font-medium">Analysis Run</Label>
            <Select value={selectedRun?.id || ""} onValueChange={(v) => setSelectedRunId(v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder="Select an analysis run..." />
              </SelectTrigger>
              <SelectContent>
                {analysisRuns.map((run) => (
                  <SelectItem key={run.id} value={run.id}>
                    {run.name} — {new Date(run.created_at).toLocaleDateString()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-heading font-medium">Epoch Resolution</Label>
            <Badge variant="secondary" className="font-body">{epochMs / 1000}s</Badge>
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-heading font-medium">Normalization</Label>
            <Badge variant="outline" className="font-body capitalize">{normalization}</Badge>
          </div>
        </div>
      </Card>

      {/* Cascade Summary Banner */}
      {cascade && cascade.cascadeOrder.length > 0 && (
        <Card className="glass-panel p-4 border-accent/30 bg-accent/5">
          <div className="flex items-start gap-3">
            <Zap className="w-5 h-5 text-accent mt-0.5 flex-shrink-0" />
            <div className="space-y-2 flex-1">
              <h3 className="font-heading text-sm font-bold">Synchrony Cascade Detected</h3>
              
              {/* Cascade order visualization */}
              <div className="flex items-center gap-2 flex-wrap">
                {cascade.cascadeOrder.map((mod, i) => {
                  const Icon = MODALITY_ICONS[mod] || Info;
                  const onset = cascade.onsets.find((o: OnsetResult) => o.modality === mod);
                  return (
                    <div key={mod} className="flex items-center gap-1.5">
                      <div className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-background border" style={{ borderColor: MODALITY_COLORS[mod] }}>
                        <Icon className="w-3.5 h-3.5" style={{ color: MODALITY_COLORS[mod] }} />
                        <span className="text-xs font-semibold capitalize">{mod}</span>
                        {onset?.onsetTimeSec != null && (
                          <Badge variant="secondary" className="text-[9px] ml-1">{onset.onsetTimeSec.toFixed(0)}s</Badge>
                        )}
                        {onset?.onsetTimeSec == null && (
                          <Badge variant="outline" className="text-[9px] ml-1 text-muted-foreground">—</Badge>
                        )}
                      </div>
                      {i < cascade.cascadeOrder.length - 1 && (
                        <ArrowRight className="w-3.5 h-3.5 text-accent" />
                      )}
                    </div>
                  );
                })}
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">{cascade.summary}</p>

              {/* Warnings */}
              {cascade.warnings && cascade.warnings.length > 0 && (
                <div className="space-y-1 mt-2">
                  {cascade.warnings.map((w: string, i: number) => (
                    <div key={i} className="flex items-start gap-1.5 text-[10px] text-amber-600">
                      <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* Stream Info */}
      {report?.streams && report.streams.length > 0 && (
        <Card className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-4 h-4 text-accent" />
            <h3 className="font-heading text-sm font-semibold">Aligned Streams</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {report.streams.map((s: any, i: number) => {
              const Icon = MODALITY_ICONS[s.modality] || Info;
              return (
                <div key={i} className="bg-muted/30 rounded-lg p-3 space-y-1">
                  <div className="flex items-center gap-1.5">
                    <Icon className="w-3.5 h-3.5" style={{ color: MODALITY_COLORS[s.modality] }} />
                    <span className="text-xs font-heading font-semibold">{s.name}</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] capitalize">{s.modality}</Badge>
                  <div className="text-[10px] text-muted-foreground mt-1">
                    <p>{s.rawSamples} samples @ {s.sampleRateHz}Hz</p>
                    <p>{s.epochedPoints} epochs</p>
                    {s.offsetApplied !== 0 && (
                      <p className="text-accent">Offset: {s.offsetApplied > 0 ? "+" : ""}{s.offsetApplied}ms</p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Main Chart */}
      <Card className="glass-panel p-4">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={{ stroke: "hsl(var(--border))" }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              label={{
                value: normalization === "zscore" ? "Z-score" : "Sync [0–1]",
                angle: -90,
                position: "insideLeft",
                style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "var(--radius)",
                fontSize: 11,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine y={0} stroke="hsl(var(--border))" strokeDasharray="2 2" />

            {/* Onset markers */}
            {cascade?.onsets
              .filter((o: OnsetResult) => o.onsetEpoch != null)
              .map((o: OnsetResult) => (
                <ReferenceLine
                  key={`onset-${o.modality}`}
                  x={chartData[o.onsetEpoch!]?.time}
                  stroke={MODALITY_COLORS[o.modality] || "hsl(var(--accent))"}
                  strokeDasharray="4 2"
                  strokeWidth={1.5}
                  label={{
                    value: `${(MODALITY_LABELS[o.modality] || o.modality)} onset`,
                    position: "top",
                    fontSize: 9,
                    fill: MODALITY_COLORS[o.modality],
                  }}
                />
              ))}

            {modalities.map((mod) => (
              <Line
                key={mod}
                type="monotone"
                dataKey={mod}
                name={`${(MODALITY_LABELS[mod] || mod)} Sync`}
                stroke={MODALITY_COLORS[mod] || "hsl(var(--accent))"}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Lead-Lag Matrix */}
      {cascade && cascade.leadLagMatrix.length > 0 && (
        <Card className="glass-panel p-4 space-y-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-accent" />
            <h3 className="font-heading text-sm font-semibold">Lead-Lag Cross-Correlation</h3>
          </div>
          <p className="text-[10px] text-muted-foreground">
            Pairwise temporal relationships between modality synchrony timeseries (Boker et al., 2002).
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-3 font-heading">Leader</th>
                  <th className="text-left py-2 px-3 font-heading">Follower</th>
                  <th className="text-center py-2 px-3 font-heading">Lag</th>
                  <th className="text-center py-2 px-3 font-heading">r</th>
                </tr>
              </thead>
              <tbody>
                {cascade.leadLagMatrix.map((ll: LeadLagResult, i: number) => (
                  <tr key={i} className="border-b border-border/50">
                    <td className="py-2 px-3">
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: MODALITY_COLORS[ll.from] }} />
                        <span className="capitalize font-medium">{ll.from}</span>
                      </div>
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex items-center gap-1.5">
                        <ArrowRight className="w-3 h-3 text-muted-foreground" />
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: MODALITY_COLORS[ll.to] }} />
                        <span className="capitalize">{ll.to}</span>
                      </div>
                    </td>
                    <td className="py-2 px-3 text-center font-mono">
                      {ll.optimalLagSec}s
                      <span className="text-muted-foreground ml-1">({ll.optimalLagEpochs} ep)</span>
                    </td>
                    <td className="py-2 px-3 text-center font-mono">
                      <Badge variant={ll.peakCorrelation > 0.5 ? "default" : "secondary"} className="text-[10px]">
                        {ll.peakCorrelation.toFixed(3)}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Granger Causality */}
      {cascade && cascade.grangerResults.length > 0 && (
        <Card className="glass-panel p-4 space-y-3">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-accent" />
            <h3 className="font-heading text-sm font-semibold">Directional Influence (Granger Causality)</h3>
          </div>
           <p className="text-[10px] text-muted-foreground">
             Tests whether past values of one modality's synchrony improve prediction of another (Granger, 1969). 
             P-values are Bonferroni-corrected for {cascade.grangerResults.length} comparisons. η² = partial eta-squared effect size.
           </p>
           <div className="overflow-x-auto">
             <table className="w-full text-xs">
               <thead>
                 <tr className="border-b border-border">
                   <th className="text-left py-2 px-3 font-heading">Cause</th>
                   <th className="text-left py-2 px-3 font-heading">Effect</th>
                   <th className="text-center py-2 px-3 font-heading">F</th>
                   <th className="text-center py-2 px-3 font-heading">p (corr)</th>
                   <th className="text-center py-2 px-3 font-heading">η²</th>
                   <th className="text-center py-2 px-3 font-heading">Result</th>
                 </tr>
               </thead>
               <tbody>
                 {cascade.grangerResults
                   .sort((a: GrangerResult, b: GrangerResult) => b.fStatistic - a.fStatistic)
                   .map((g: GrangerResult, i: number) => (
                     <tr key={i} className="border-b border-border/50">
                       <td className="py-2 px-3">
                         <div className="flex items-center gap-1.5">
                           <div className="w-2 h-2 rounded-full" style={{ backgroundColor: MODALITY_COLORS[g.cause] }} />
                           <span className="capitalize font-medium">{g.cause}</span>
                         </div>
                       </td>
                       <td className="py-2 px-3">
                         <div className="flex items-center gap-1.5">
                           <ArrowRight className="w-3 h-3 text-muted-foreground" />
                           <div className="w-2 h-2 rounded-full" style={{ backgroundColor: MODALITY_COLORS[g.effect] }} />
                           <span className="capitalize">{g.effect}</span>
                         </div>
                       </td>
                       <td className="py-2 px-3 text-center font-mono">{g.fStatistic.toFixed(2)}</td>
                       <td className="py-2 px-3 text-center font-mono">
                         {g.pValueCorrected != null ? (
                           <span className={g.pValueCorrected < 0.05 ? "text-accent font-semibold" : ""}>
                             {g.pValueCorrected < 0.001 ? "<.001" : g.pValueCorrected.toFixed(3)}
                           </span>
                         ) : "—"}
                       </td>
                       <td className="py-2 px-3 text-center font-mono">
                         {g.effectSize != null ? g.effectSize.toFixed(3) : "—"}
                       </td>
                       <td className="py-2 px-3 text-center">
                         <div className="flex items-center justify-center gap-1">
                           {g.direction === "causes" ? (
                             <Badge className="text-[10px] bg-accent/20 text-accent border-accent/30">Causes</Badge>
                           ) : (
                             <Badge variant="outline" className="text-[10px] text-muted-foreground">No effect</Badge>
                           )}
                           {g.underpowered && (
                             <Badge variant="outline" className="text-[9px] text-amber-500 border-amber-500/30">Low n</Badge>
                           )}
                         </div>
                       </td>
                     </tr>
                   ))}
               </tbody>
             </table>
           </div>
         </Card>
       )}

       {/* Onset Sensitivity Analysis */}
       {cascade && cascade.sensitivityAnalysis && cascade.sensitivityAnalysis.length > 0 && (
         <Card className="glass-panel p-4 space-y-3">
           <div className="flex items-center gap-2">
             <TrendingUp className="w-4 h-4 text-accent" />
             <h3 className="font-heading text-sm font-semibold">Onset Threshold Sensitivity</h3>
             {cascade.cascadeStability != null && (
               <Badge variant={cascade.cascadeStability >= 0.7 ? "default" : "secondary"} className="text-[10px]">
                 Stability: {(cascade.cascadeStability * 100).toFixed(0)}%
               </Badge>
             )}
           </div>
           <p className="text-[10px] text-muted-foreground">
             Cascade order at different onset thresholds (0.25σ–1.5σ). High stability means the leading modality is robust to threshold choice.
           </p>
           <div className="overflow-x-auto">
             <table className="w-full text-xs">
               <thead>
                 <tr className="border-b border-border">
                   <th className="text-left py-2 px-3 font-heading">Threshold</th>
                   <th className="text-left py-2 px-3 font-heading">Cascade Order</th>
                 </tr>
               </thead>
               <tbody>
                 {cascade.sensitivityAnalysis.map((s: SensitivityResult, i: number) => (
                   <tr key={i} className={`border-b border-border/50 ${s.threshold === cascade.thresholdSigma ? "bg-accent/5" : ""}`}>
                     <td className="py-2 px-3 font-mono">
                       {s.threshold.toFixed(2)}σ
                       {s.threshold === cascade.thresholdSigma && (
                         <span className="text-accent ml-1">←</span>
                       )}
                     </td>
                     <td className="py-2 px-3">
                       <div className="flex items-center gap-1 flex-wrap">
                         {s.cascadeOrder.map((mod, j) => (
                           <span key={j} className="flex items-center gap-0.5">
                             {j > 0 && <ArrowRight className="w-2.5 h-2.5 text-muted-foreground" />}
                             <span className="capitalize" style={{ color: MODALITY_COLORS[mod] || "inherit" }}>{mod}</span>
                           </span>
                         ))}
                         {s.cascadeOrder.length === 0 && (
                           <span className="text-muted-foreground italic">No modality crossed threshold</span>
                         )}
                       </div>
                     </td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
         </Card>
       )}

      {/* Modality Summary Cards with onset info */}
      {modalities.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {modalities.map((mod) => {
            const Icon = MODALITY_ICONS[mod] || Info;
            const values = chartData.map((d: any) => d[mod]).filter((v: any) => v != null) as number[];
            if (values.length === 0) return null;
            const peak = Math.max(...values);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const streamInfo = report?.streams?.find((s: any) => s.modality === mod);
            const onset = cascade?.onsets?.find((o: OnsetResult) => o.modality === mod);

            return (
              <Card key={mod} className="glass-panel p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4" style={{ color: MODALITY_COLORS[mod] }} />
                    <span className="font-heading text-sm font-semibold capitalize">{mod}</span>
                  </div>
                  {onset?.onsetTimeSec != null && cascade?.cascadeOrder?.[0] === mod && (
                    <Badge className="text-[9px] bg-accent/20 text-accent">Leader</Badge>
                  )}
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs font-body">
                  <div>
                    <p className="text-muted-foreground">Peak</p>
                    <p className="font-semibold">{peak.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Mean</p>
                    <p className="font-semibold">{mean.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Onset</p>
                    <p className="font-semibold">
                      {onset?.onsetTimeSec != null ? `${onset.onsetTimeSec.toFixed(0)}s` : "—"}
                    </p>
                  </div>
                </div>
                {streamInfo && (
                  <div className="text-[10px] text-muted-foreground">
                    {streamInfo.sampleRateHz}Hz native → {epochMs / 1000}s epoch
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
