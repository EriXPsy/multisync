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
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { generateDemoData, type TimelineConfig, type ModalityStream } from "@/lib/synchrony-data";
import { Brain, Eye, Heart, Users, Info, ArrowRight } from "lucide-react";

const MODALITY_ICONS = {
  neural: Brain,
  behavioral: Eye,
  bio: Heart,
  psycho: Users,
};

const MODALITY_COLORS = {
  neural: "hsl(262, 60%, 55%)",
  behavioral: "hsl(185, 55%, 40%)",
  bio: "hsl(340, 60%, 55%)",
  psycho: "hsl(35, 80%, 55%)",
};

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function detectOnsetEpoch(data: { value: number }[], threshold: number = 0.5): number {
  for (let i = 0; i < data.length; i++) {
    if (data[i].value > threshold) return i;
  }
  return data.length - 1;
}

export function UnifiedTimeline() {
  const [epochMs, setEpochMs] = useState(10000); // 10 second default
  const [totalDurationMs] = useState(300000); // 5 minutes
  const [normalization, setNormalization] = useState<"zscore" | "minmax">("zscore");

  const config: TimelineConfig = { commonEpochMs: epochMs, totalDurationMs };
  const streams = useMemo(() => generateDemoData(config), [epochMs, totalDurationMs]);

  // Build chart data
  const chartData = useMemo(() => {
    const numEpochs = Math.floor(totalDurationMs / epochMs);
    return Array.from({ length: numEpochs }, (_, i) => {
      const point: Record<string, number | string> = {
        epoch: i,
        time: formatTime(i * epochMs),
        timeMs: i * epochMs,
      };
      streams.forEach((s) => {
        const dp = s.compositeScore[i];
        if (dp) {
          point[s.modality] = normalization === "zscore" ? dp.value : dp.rawValue;
          point[`${s.modality}_conf`] = dp.confidence;
        }
      });
      return point;
    });
  }, [streams, epochMs, totalDurationMs, normalization]);

  // Detect cascade onsets
  const onsets = useMemo(() => {
    return streams.map((s) => ({
      modality: s.modality,
      label: s.label,
      onsetEpoch: detectOnsetEpoch(s.compositeScore),
      onsetTimeMs: detectOnsetEpoch(s.compositeScore) * epochMs,
    })).sort((a, b) => a.onsetEpoch - b.onsetEpoch);
  }, [streams, epochMs]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="font-heading text-2xl font-bold text-foreground">
          Unified Multimodal Timeline
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          All synchrony modalities epoch-aggregated to a common resolution, revealing temporal cascade patterns.
        </p>
      </div>

      {/* Configuration Bar */}
      <Card className="glass-panel p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <Label className="text-xs font-heading font-medium">
              Common Epoch Resolution
            </Label>
            <div className="flex items-center gap-3">
              <Slider
                value={[epochMs]}
                onValueChange={([v]) => setEpochMs(v)}
                min={1000}
                max={30000}
                step={1000}
                className="flex-1"
              />
              <Badge variant="secondary" className="min-w-[60px] text-center font-body">
                {epochMs / 1000}s
              </Badge>
            </div>
            <p className="text-[10px] text-muted-foreground">
              Each stream computes synchrony at native window, then aggregates to this epoch.
            </p>
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-heading font-medium">Normalization</Label>
            <Select value={normalization} onValueChange={(v) => setNormalization(v as "zscore" | "minmax")}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="zscore">Z-Score (recommended)</SelectItem>
                <SelectItem value="minmax">Min-Max [0,1]</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-[10px] text-muted-foreground">
              Z-scoring per modality enables cross-modal comparison on a unified scale.
            </p>
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-heading font-medium">Duration</Label>
            <Badge variant="outline" className="font-body">{totalDurationMs / 60000} min</Badge>
            <p className="text-[10px] text-muted-foreground">
              {Math.floor(totalDurationMs / epochMs)} epochs total at current resolution.
            </p>
          </div>
        </div>
      </Card>

      {/* Cascade Detection */}
      <Card className="glass-panel p-4">
        <div className="flex items-center gap-2 mb-3">
          <Info className="w-4 h-4 text-accent" />
          <h3 className="font-heading text-sm font-semibold">Synchrony Cascade Detection</h3>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {onsets.map((onset, i) => {
            const Icon = MODALITY_ICONS[onset.modality as keyof typeof MODALITY_ICONS];
            return (
              <div key={onset.modality} className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-muted">
                  <Icon className="w-3.5 h-3.5" style={{ color: MODALITY_COLORS[onset.modality as keyof typeof MODALITY_COLORS] }} />
                  <span className="text-xs font-medium font-heading">{onset.label}</span>
                  <Badge variant="secondary" className="text-[10px] font-body">
                    {formatTime(onset.onsetTimeMs)}
                  </Badge>
                </div>
                {i < onsets.length - 1 && (
                  <ArrowRight className="w-3.5 h-3.5 text-muted-foreground" />
                )}
              </div>
            );
          })}
        </div>
        <p className="text-[10px] text-muted-foreground mt-2">
          Onset detected when composite z-score exceeds threshold (0.5σ). Neural synchrony typically emerges first (ms-scale), followed by behavioral (100ms-1s), then bio (seconds), then psycho (minutes).
        </p>
      </Card>

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
                fontFamily: "Manrope",
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, fontFamily: "Sora" }}
            />
            <ReferenceLine y={0} stroke="hsl(var(--border))" strokeDasharray="2 2" />
            
            {streams.map((s) => (
              <Line
                key={s.modality}
                type="monotone"
                dataKey={s.modality}
                name={s.label}
                stroke={MODALITY_COLORS[s.modality as keyof typeof MODALITY_COLORS]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Modality Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {streams.map((s) => {
          const Icon = MODALITY_ICONS[s.modality as keyof typeof MODALITY_ICONS];
          const peak = Math.max(...s.compositeScore.map((d) => d.rawValue));
          const mean = s.compositeScore.reduce((a, d) => a + d.rawValue, 0) / s.compositeScore.length;
          
          return (
            <Card key={s.modality} className="glass-panel p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icon
                    className="w-4 h-4"
                    style={{ color: MODALITY_COLORS[s.modality as keyof typeof MODALITY_COLORS] }}
                  />
                  <span className="font-heading text-sm font-semibold">{s.label}</span>
                </div>
                <Badge
                  className="text-[10px]"
                  style={{
                    backgroundColor: `${MODALITY_COLORS[s.modality as keyof typeof MODALITY_COLORS]}20`,
                    color: MODALITY_COLORS[s.modality as keyof typeof MODALITY_COLORS],
                    border: "none",
                  }}
                >
                  {s.indices.length} indices
                </Badge>
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
              <div className="text-[10px] text-muted-foreground">
                Native: {s.indices[0]?.nativeResolutionMs}ms → Epoch: {epochMs / 1000}s
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
