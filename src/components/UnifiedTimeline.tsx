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
import { Brain, Eye, Heart, Users, Info, AlertTriangle, BarChart3 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

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
    return analysisRuns[0]; // default to latest
  }, [analysisRuns, selectedRunId]);

  const chartData = useMemo(() => {
    if (!selectedRun?.results) return [];
    return selectedRun.results as any[];
  }, [selectedRun]);

  const report = useMemo(() => {
    if (!selectedRun?.alignment_report) return null;
    return selectedRun.alignment_report as any;
  }, [selectedRun]);

  const modalities = useMemo(() => {
    if (!chartData || chartData.length === 0) return [];
    const keys = Object.keys(chartData[0]).filter(
      (k) => !["epoch", "time", "timeMs"].includes(k) && !k.endsWith("_conf")
    );
    return keys;
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
          Showing real analysis results — epoch-aggregated synchrony from the pipeline.
        </p>
      </div>

      {/* Run Selector */}
      <Card className="glass-panel p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label className="text-xs font-heading font-medium">Analysis Run</Label>
            <Select
              value={selectedRun?.id || ""}
              onValueChange={(v) => setSelectedRunId(v)}
            >
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

      {/* Stream Info from alignment report */}
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

            {modalities.map((mod) => (
              <Line
                key={mod}
                type="monotone"
                dataKey={mod}
                name={`${mod.charAt(0).toUpperCase() + mod.slice(1)} Sync`}
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

      {/* Modality Summary Cards */}
      {modalities.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {modalities.map((mod) => {
            const Icon = MODALITY_ICONS[mod] || Info;
            const values = chartData.map((d: any) => d[mod]).filter((v: any) => v != null) as number[];
            if (values.length === 0) return null;
            const peak = Math.max(...values);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const streamInfo = report?.streams?.find((s: any) => s.modality === mod);

            return (
              <Card key={mod} className="glass-panel p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4" style={{ color: MODALITY_COLORS[mod] }} />
                    <span className="font-heading text-sm font-semibold capitalize">{mod}</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs font-body">
                  <div>
                    <p className="text-muted-foreground">Peak</p>
                    <p className="font-semibold">{peak.toFixed(3)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Mean</p>
                    <p className="font-semibold">{mean.toFixed(3)}</p>
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
