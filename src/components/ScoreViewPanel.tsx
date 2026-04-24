/**
 * Score View Panel — Context Annotation Track
 *
 * Music analogy: Like a musical score where different "movements" (context phases)
 * are marked above the staff. Researchers can annotate segments of the interaction
 * (e.g., "Free Conversation", "Joint Task", "Rest") to analyze how synchrony
 * patterns change across contextual phases.
 *
 * Gordon et al. (2024): "A Theory of Flexible Multimodal Synchrony" argues that
 * context creates competing pulls toward synchrony vs. segregation. Score View
 * makes these contextual phases visible and analyzable.
 */

import { useState, useCallback, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Music,
  Plus,
  X,
  Pencil,
  ChevronDown,
} from "lucide-react";

export interface ContextSegment {
  id: string;
  label: string;
  startEpoch: number;
  endEpoch: number;
  color: string;
  description?: string;
}

const PRESET_COLORS = [
  "hsl(262, 60%, 55%)",  // purple
  "hsl(185, 55%, 40%)",  // teal
  "hsl(340, 60%, 55%)",  // rose
  "hsl(35, 80%, 55%)",   // amber
  "hsl(120, 40%, 40%)",  // green
  "hsl(210, 60%, 50%)",  // blue
  "hsl(0, 60%, 50%)",    // red
  "hsl(280, 50%, 50%)",  // violet
];

const PRESET_LABELS = [
  "Free Conversation",
  "Joint Task",
  "Rest / Baseline",
  "Conflict / Negotiation",
  "Emotional Sharing",
  "Cooperative Problem",
  "Transition",
  "Debrief",
];

interface ScoreViewPanelProps {
  /** Chart data with epoch index */
  chartData: Record<string, any>[];
  /** Milliseconds per epoch */
  epochMs: number;
  /** Existing segments (controlled) */
  segments: ContextSegment[];
  /** Callback when segments change */
  onSegmentsChange: (segments: ContextSegment[]) => void;
}

const ScoreViewPanel: React.FC<ScoreViewPanelProps> = ({
  chartData,
  epochMs,
  segments,
  onSegmentsChange,
}) => {
  const [editingSegment, setEditingSegment] = useState<ContextSegment | null>(null);
  const [isAdding, setIsAdding] = useState(false);
  const [showPresets, setShowPresets] = useState(false);

  const totalEpochs = chartData.length;
  const timeStartMs = 0;
  const timeEndMs = totalEpochs * epochMs;

  const addSegment = useCallback(
    (label: string, color: string) => {
      // Find the largest gap between existing segments
      const sorted = [...segments].sort((a, b) => a.startEpoch - b.startEpoch);
      let bestStart = 0;
      let bestEnd = totalEpochs;
      let bestGap = 0;

      // Before first segment
      if (sorted.length > 0 && sorted[0].startEpoch > 1) {
        bestGap = sorted[0].startEpoch - 0;
        bestStart = 0;
        bestEnd = sorted[0].startEpoch;
      }

      // Between segments
      for (let i = 0; i < sorted.length - 1; i++) {
        const gap = sorted[i + 1].startEpoch - sorted[i].endEpoch;
        if (gap > bestGap) {
          bestGap = gap;
          bestStart = sorted[i].endEpoch;
          bestEnd = sorted[i + 1].startEpoch;
        }
      }

      // After last segment
      if (sorted.length > 0) {
        const afterGap = totalEpochs - sorted[sorted.length - 1].endEpoch;
        if (afterGap > bestGap) {
          bestStart = sorted[sorted.length - 1].endEpoch;
          bestEnd = totalEpochs;
        }
      }

      // Default to full range if no segments exist
      if (segments.length === 0) {
        bestStart = 0;
        bestEnd = totalEpochs;
      }

      const newSegment: ContextSegment = {
        id: `seg_${Date.now()}`,
        label,
        startEpoch: bestStart,
        endEpoch: bestEnd,
        color,
      };
      onSegmentsChange([...segments, newSegment]);
      setEditingSegment(newSegment);
      setIsAdding(false);
      setShowPresets(false);
    },
    [segments, totalEpochs, onSegmentsChange]
  );

  const removeSegment = useCallback(
    (id: string) => {
      onSegmentsChange(segments.filter((s) => s.id !== id));
      if (editingSegment?.id === id) setEditingSegment(null);
    },
    [segments, editingSegment, onSegmentsChange]
  );

  const updateSegment = useCallback(
    (id: string, updates: Partial<ContextSegment>) => {
      onSegmentsChange(
        segments.map((s) => (s.id === id ? { ...s, ...updates } : s))
      );
      if (editingSegment?.id === id) {
        setEditingSegment({ ...editingSegment, ...updates });
      }
    },
    [segments, editingSegment, onSegmentsChange]
  );

  const formatTime = (epoch: number) => {
    const sec = (epoch * epochMs) / 1000;
    if (sec < 60) return `${sec.toFixed(0)}s`;
    const min = Math.floor(sec / 60);
    const remSec = (sec % 60).toFixed(0);
    return `${min}m${remSec}s`;
  };

  // Compute per-segment statistics from chartData
  const getSegmentStats = (seg: ContextSegment) => {
    const modalityKeys = Object.keys(chartData[0] || {}).filter(
      (k) => !["epoch", "time"].includes(k) && typeof chartData[0][k] === "number"
    );
    const stats: Record<string, { mean: number; max: number; min: number }> = {};

    for (const mod of modalityKeys) {
      const values = chartData
        .slice(seg.startEpoch, seg.endEpoch)
        .map((d) => d[mod])
        .filter((v) => v !== null && v !== undefined);
      if (values.length > 0) {
        stats[mod] = {
          mean: parseFloat((values.reduce((a, b) => a + b, 0) / values.length).toFixed(3)),
          max: parseFloat(Math.max(...values).toFixed(3)),
          min: parseFloat(Math.min(...values).toFixed(3)),
        };
      }
    }
    return stats;
  };

  return (
    <Card className="glass-panel p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Music className="w-4 h-4 text-purple-400" />
          <h3 className="font-heading text-sm font-semibold">Score View</h3>
          <Badge variant="outline" className="text-[10px]">
            Context Phases
          </Badge>
        </div>
        <div className="flex items-center gap-1">
          {!isAdding && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-[10px]"
              onClick={() => setShowPresets(!showPresets)}
            >
              <Plus className="w-3 h-3 mr-1" /> Add Phase
            </Button>
          )}
        </div>
      </div>

      {/* Preset label picker */}
      {showPresets && (
        <div className="p-2 bg-muted/30 rounded-lg space-y-2">
          <p className="text-[10px] text-muted-foreground font-medium">Choose a phase type:</p>
          <div className="flex flex-wrap gap-1">
            {PRESET_LABELS.map((label, i) => (
              <button
                key={label}
                className="text-[10px] px-2 py-1 rounded-md bg-background border border-border hover:bg-accent/50 transition-colors"
                onClick={() => addSegment(label, PRESET_COLORS[i % PRESET_COLORS.length])}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground">
            or{" "}
            <button
              className="text-accent underline"
              onClick={() => {
                addSegment("Custom Phase", PRESET_COLORS[segments.length % PRESET_COLORS.length]);
              }}
            >
              add custom phase
            </button>
          </p>
        </div>
      )}

      {/* Visual track */}
      <div className="relative">
        {/* Timeline ruler */}
        <div className="flex justify-between text-[9px] text-muted-foreground mb-1 px-1">
          <span>0s</span>
          <span>{formatTime(Math.floor(totalEpochs / 4))}</span>
          <span>{formatTime(Math.floor(totalEpochs / 2))}</span>
          <span>{formatTime(Math.floor((3 * totalEpochs) / 4))}</span>
          <span>{formatTime(totalEpochs)}</span>
        </div>

        {/* Segment track */}
        <div className="h-10 bg-muted/20 rounded-md relative overflow-hidden border border-border/50">
          {segments.length === 0 ? (
            <div className="flex items-center justify-center h-full text-[10px] text-muted-foreground/60">
              Click "Add Phase" to annotate context segments
            </div>
          ) : (
            segments.map((seg) => {
              const left = totalEpochs > 0 ? (seg.startEpoch / totalEpochs) * 100 : 0;
              const width =
                totalEpochs > 0
                  ? ((seg.endEpoch - seg.startEpoch) / totalEpochs) * 100
                  : 100;
              return (
                <div
                  key={seg.id}
                  className="absolute top-0 bottom-0 flex items-center justify-center cursor-pointer group"
                  style={{
                    left: `${left}%`,
                    width: `${Math.max(width, 2)}%`,
                    backgroundColor: seg.color + "30",
                    borderLeft: `3px solid ${seg.color}`,
                    borderRight: `3px solid ${seg.color}`,
                  }}
                  onClick={() => setEditingSegment(editingSegment?.id === seg.id ? null : seg)}
                >
                  <span className="text-[9px] font-medium truncate px-1 text-foreground">
                    {seg.label}
                  </span>
                  {/* Delete button */}
                  <button
                    className="absolute top-0.5 right-0.5 w-3.5 h-3.5 rounded-full bg-red-500/80 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeSegment(seg.id);
                    }}
                  >
                    <X className="w-2 h-2" />
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Unannotated portions are implicitly "baseline" */}
        {segments.length > 0 && (
          <p className="text-[9px] text-muted-foreground/50 mt-1 text-center">
            Unannotated regions are treated as undifferentiated baseline.
          </p>
        )}
      </div>

      {/* Editing panel for selected segment */}
      {editingSegment && (
        <div className="p-3 bg-muted/20 rounded-lg border border-border/50 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Pencil className="w-3 h-3 text-muted-foreground" />
              <span className="text-[11px] font-heading font-semibold">
                Edit: {editingSegment.label}
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-[10px]"
              onClick={() => setEditingSegment(null)}
            >
              Done
            </Button>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[9px] text-muted-foreground">Label</label>
              <Input
                className="h-7 text-[10px] mt-0.5"
                value={editingSegment.label}
                onChange={(e) =>
                  updateSegment(editingSegment.id, { label: e.target.value })
                }
              />
            </div>
            <div>
              <label className="text-[9px] text-muted-foreground">Description (optional)</label>
              <Input
                className="h-7 text-[10px] mt-0.5"
                placeholder="e.g., Joint puzzle task..."
                value={editingSegment.description || ""}
                onChange={(e) =>
                  updateSegment(editingSegment.id, { description: e.target.value })
                }
              />
            </div>
            <div>
              <label className="text-[9px] text-muted-foreground">Start (epoch)</label>
              <Input
                className="h-7 text-[10px] mt-0.5"
                type="number"
                min={0}
                max={totalEpochs - 1}
                value={editingSegment.startEpoch}
                onChange={(e) =>
                  updateSegment(editingSegment.id, {
                    startEpoch: Math.max(0, parseInt(e.target.value) || 0),
                  })
                }
              />
              <p className="text-[8px] text-muted-foreground/60 mt-0.5">
                = {formatTime(editingSegment.startEpoch)}
              </p>
            </div>
            <div>
              <label className="text-[9px] text-muted-foreground">End (epoch)</label>
              <Input
                className="h-7 text-[10px] mt-0.5"
                type="number"
                min={1}
                max={totalEpochs}
                value={editingSegment.endEpoch}
                onChange={(e) =>
                  updateSegment(editingSegment.id, {
                    endEpoch: Math.min(totalEpochs, parseInt(e.target.value) || 1),
                  })
                }
              />
              <p className="text-[8px] text-muted-foreground/60 mt-0.5">
                = {formatTime(editingSegment.endEpoch)} ({formatTime(editingSegment.endEpoch - editingSegment.startEpoch)} duration)
              </p>
            </div>
          </div>

          {/* Per-segment stats */}
          {editingSegment.endEpoch > editingSegment.startEpoch && chartData.length > 0 && (
            <div>
              <p className="text-[9px] text-muted-foreground font-medium mb-1">
                Mean synchrony within this phase:
              </p>
              <div className="flex gap-2 flex-wrap">
                {Object.entries(getSegmentStats(editingSegment)).map(([mod, stats]) => (
                  <span
                    key={mod}
                    className="text-[9px] px-1.5 py-0.5 rounded bg-muted/40"
                  >
                    <span className="capitalize font-medium">{mod}</span>:{" "}
                    <span className="font-mono">{stats.mean.toFixed(2)}</span>
                    <span className="text-muted-foreground ml-1">
                      [{stats.min.toFixed(1)}, {stats.max.toFixed(1)}]
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Segment summary table */}
      {segments.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-1 pr-2 font-heading font-semibold">Phase</th>
                <th className="text-right py-1 px-1 font-heading">Start</th>
                <th className="text-right py-1 px-1 font-heading">End</th>
                <th className="text-right py-1 px-1 font-heading">Duration</th>
                <th className="text-right py-1 px-1 font-heading">Epochs</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((seg) => (
                <tr key={seg.id} className="border-b border-border/30">
                  <td className="py-1 pr-2">
                    <span
                      className="inline-block w-2 h-2 rounded-full mr-1"
                      style={{ backgroundColor: seg.color }}
                    />
                    <span className="font-medium">{seg.label}</span>
                  </td>
                  <td className="text-right py-1 px-1 font-mono">
                    {formatTime(seg.startEpoch)}
                  </td>
                  <td className="text-right py-1 px-1 font-mono">
                    {formatTime(seg.endEpoch)}
                  </td>
                  <td className="text-right py-1 px-1 font-mono">
                    {formatTime(seg.endEpoch - seg.startEpoch)}
                  </td>
                  <td className="text-right py-1 px-1 font-mono">
                    {seg.endEpoch - seg.startEpoch}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Theoretical note */}
      <p className="text-[9px] text-muted-foreground italic leading-relaxed">
        Inspired by Gordon et al. (2025, Psychological Review): context creates competing
        pulls toward synchrony vs. segregation. Annotating phases lets you test whether
        synchrony levels, cascade patterns, and modality dominance shift across different
        interaction contexts — the key claim of flexible multimodal synchrony theory.
      </p>
    </Card>
  );
};

export default ScoreViewPanel;
