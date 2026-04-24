/**
 * Cascade Map Panel — Modality Synchrony Cascade Directed Graph
 *
 * Visualizes the cascade of synchrony onset across modalities, showing
 * which modality "leads" (the prelude signal) and how synchrony propagates.
 *
 * Music analogy: Like seeing which instrument starts playing first and how
 * others join in — the "melody" of synchrony emergence.
 *
 * Data sources:
 * - Onset detection from dynamic-features.ts (onsetLatencySec per modality)
 * - Lead-lag cross-correlation from cascade-analysis.ts
 * - Granger causality from cascade-analysis.ts
 */

import { useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  GitBranch,
  ArrowRight,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import type {
  OnsetResult,
  LeadLagResult,
  GrangerResult,
  SensitivityResult,
} from "@/lib/cascade-analysis";
import type { DynamicFeatureReport } from "@/lib/dynamic-features";

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

interface CascadeMapPanelProps {
  /** Onset results from cascade analysis */
  onsets: OnsetResult[];
  /** Cascade order (sorted by onset time) */
  cascadeOrder: string[];
  /** Lead-lag cross-correlation results */
  leadLagMatrix: LeadLagResult[];
  /** Granger causality results */
  grangerResults: GrangerResult[];
  /** Sensitivity analysis across thresholds */
  sensitivityAnalysis: SensitivityResult[];
  /** Cascade stability score (0-1) */
  cascadeStability: number;
  /** Dynamic features for enrichment */
  dynamicReport: DynamicFeatureReport | null;
}

const CascadeMapPanel: React.FC<CascadeMapPanelProps> = ({
  onsets,
  cascadeOrder,
  leadLagMatrix,
  grangerResults,
  sensitivityAnalysis,
  cascadeStability,
  dynamicReport,
}) => {
  // Build the cascade chain with timing info
  const chain = useMemo(() => {
    if (onsets.length === 0) return [];

    return onsets
      .filter((o) => o.onsetEpoch !== null)
      .sort((a, b) => (a.onsetEpoch ?? Infinity) - (b.onsetEpoch ?? Infinity))
      .map((o) => ({
        modality: o.modality,
        onsetEpoch: o.onsetEpoch!,
        onsetTimeSec: o.onsetTimeSec!,
        peakValue: o.peakValue,
        meanValue: o.meanValue,
        color: MODALITY_COLORS[o.modality] || "hsl(var(--accent))",
        label: MODALITY_LABELS[o.modality] || o.modality,
      }));
  }, [onsets]);

  // Compute delays between consecutive modalities
  const delays = useMemo(() => {
    const result: { from: string; to: string; delaySec: number; delayEpochs: number }[] = [];
    for (let i = 0; i < chain.length - 1; i++) {
      const delayEpochs = chain[i + 1].onsetEpoch - chain[i].onsetEpoch;
      const delaySec = chain[i + 1].onsetTimeSec - chain[i].onsetTimeSec;
      result.push({
        from: chain[i].modality,
        to: chain[i + 1].modality,
        delaySec: parseFloat(delaySec.toFixed(1)),
        delayEpochs,
      });
    }
    return result;
  }, [chain]);

  // Significant Granger results (for showing directional influence)
  const significantGranger = useMemo(
    () => grangerResults.filter((g) => g.direction === "causes"),
    [grangerResults]
  );

  // Strongest lead-lag pairs
  const strongLeadLags = useMemo(
    () =>
      [...leadLagMatrix]
        .sort((a, b) => b.peakCorrelation - a.peakCorrelation)
        .slice(0, 4),
    [leadLagMatrix]
  );

  if (chain.length === 0) {
    return (
      <Card className="glass-panel p-4 space-y-3">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-teal-400" />
          <h3 className="font-heading text-sm font-semibold">Cascade Map</h3>
          <Badge variant="outline" className="text-[10px]">
            Synchrony Onset Chain
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground">
          No modalities reached the onset threshold. Try lowering the threshold or checking your data quality.
        </p>
      </Card>
    );
  }

  return (
    <Card className="glass-panel p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-teal-400" />
          <h3 className="font-heading text-sm font-semibold">Cascade Map</h3>
          <Badge variant="outline" className="text-[10px]">
            Synchrony Onset Chain
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          {cascadeStability >= 0.7 ? (
            <span className="flex items-center gap-1 text-[10px] text-green-600">
              <ShieldCheck className="w-3 h-3" />
              Stable ({(cascadeStability * 100).toFixed(0)}%)
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] text-amber-600">
              <ShieldAlert className="w-3 h-3" />
              Unstable ({(cascadeStability * 100).toFixed(0)}%)
            </span>
          )}
        </div>
      </div>

      {/* Main cascade visualization */}
      <div className="flex items-center gap-1 overflow-x-auto py-2">
        {chain.map((node, i) => (
          <div key={node.modality} className="flex items-center gap-1 flex-shrink-0">
            {/* Node */}
            <div className="flex flex-col items-center">
              <div
                className="w-16 h-16 rounded-lg flex flex-col items-center justify-center border-2 shadow-sm"
                style={{
                  borderColor: node.color,
                  backgroundColor: node.color + "15",
                }}
              >
                <span className="text-[9px] font-heading font-bold" style={{ color: node.color }}>
                  {node.label}
                </span>
                <span className="text-[8px] text-muted-foreground font-mono mt-0.5">
                  {node.onsetTimeSec}s
                </span>
              </div>
              {/* Stats under node */}
              <div className="text-[8px] text-muted-foreground text-center mt-1 space-y-0">
                <div>Peak: <span className="font-mono">{node.peakValue.toFixed(2)}</span></div>
                {dynamicReport && (
                  <div>
                    Build:{" "}
                    <span className="font-mono">
                      {dynamicReport.perModality.find((f) => f.modality === node.modality)?.buildUpRate?.toFixed(3) || "—"}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Arrow with delay */}
            {i < chain.length - 1 && delays[i] && (
              <div className="flex flex-col items-center px-1 flex-shrink-0">
                <ArrowRight className="w-4 h-4 text-muted-foreground" />
                <span className="text-[9px] font-mono text-muted-foreground whitespace-nowrap">
                  +{delays[i].delaySec}s
                </span>
                <span className="text-[8px] text-muted-foreground/60">
                  ({delays[i].delayEpochs}e)
                </span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Cascade pattern string */}
      <div className="text-center">
        <p className="text-[10px] text-muted-foreground">
          Cascade pattern:{" "}
          <span className="font-mono font-medium text-foreground">
            {cascadeOrder.map((m) => MODALITY_LABELS[m] || m).join(" -> ")}
          </span>
        </p>
      </div>

      {/* Stability warning */}
      {cascadeStability < 0.5 && (
        <div className="flex items-start gap-1.5 p-2 rounded-md bg-amber-500/10 border border-amber-500/20">
          <AlertTriangle className="w-3 h-3 text-amber-500 mt-0.5 flex-shrink-0" />
          <p className="text-[9px] text-amber-600">
            Cascade order is unstable across thresholds ({(cascadeStability * 100).toFixed(0)}%).
            The onset sequence may be an artifact of threshold choice rather than a genuine cascade.
            Consider using Granger causality for directional inference instead.
          </p>
        </div>
      )}

      {/* Granger causality section */}
      {significantGranger.length > 0 && (
        <div className="space-y-2">
          <p className="text-[10px] font-heading font-semibold flex items-center gap-1">
            Directional Influence (Granger Causality)
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
            {significantGranger.sort((a, b) => b.fStatistic - a.fStatistic).map((g, i) => (
              <div
                key={`${g.cause}-${g.effect}`}
                className="flex items-center gap-2 p-1.5 rounded-md bg-muted/20 text-[10px]"
              >
                <span className="capitalize font-medium" style={{ color: MODALITY_COLORS[g.cause] }}>
                  {MODALITY_LABELS[g.cause] || g.cause}
                </span>
                <ArrowRight className="w-3 h-3 text-muted-foreground" />
                <span className="capitalize font-medium" style={{ color: MODALITY_COLORS[g.effect] }}>
                  {MODALITY_LABELS[g.effect] || g.effect}
                </span>
                <span className="ml-auto font-mono text-muted-foreground">
                  F={g.fStatistic.toFixed(1)}
                </span>
                <Badge variant="outline" className="text-[8px] px-1 py-0">
                  p={g.pValueCorrected < 0.001 ? "<.001" : g.pValueCorrected.toFixed(3)}
                </Badge>
                {g.underpowered && (
                  <Badge variant="secondary" className="text-[8px] px-1 py-0 text-amber-600">
                  LP
                </Badge>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Lead-lag cross-correlation */}
      {strongLeadLags.length > 0 && (
        <div className="space-y-2">
          <p className="text-[10px] font-heading font-semibold">
            Temporal Lead-Lag (Cross-Correlation)
          </p>
          <div className="space-y-1">
            {strongLeadLags.map((ll, i) => (
              <div key={`${ll.from}-${ll.to}`} className="flex items-center gap-2 text-[10px]">
                <span className="capitalize w-20 truncate" style={{ color: MODALITY_COLORS[ll.from] }}>
                  {MODALITY_LABELS[ll.from] || ll.from}
                </span>
                {ll.optimalLagEpochs > 0 ? (
                  <span className="text-muted-foreground">
                    leads {MODALITY_LABELS[ll.to] || ll.to} by {ll.optimalLagSec}s
                  </span>
                ) : (
                  <span className="text-muted-foreground">
                    synchronous with {MODALITY_LABELS[ll.to] || ll.to}
                  </span>
                )}
                <span className="ml-auto font-mono text-muted-foreground">
                  r={ll.peakCorrelation.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sensitivity analysis: cascade order across thresholds */}
      {sensitivityAnalysis.length > 0 && (
        <div className="space-y-2">
          <p className="text-[10px] font-heading font-semibold">
            Threshold Sensitivity
          </p>
          <div className="space-y-1">
            {sensitivityAnalysis.map((sa) => (
              <div key={sa.threshold} className="flex items-center gap-2 text-[9px] text-muted-foreground">
                <span className="w-10 text-right font-mono">{sa.threshold.toFixed(2)}sigma</span>
                <span className="font-mono">
                  {sa.cascadeOrder.map((m) => (MODALITY_LABELS[m] || m).charAt(0)).join(" -> ")}
                </span>
                {sa.cascadeOrder[0] === cascadeOrder[0] && (
                  <span className="text-green-600">&#10003;</span>
                )}
              </div>
            ))}
          </div>
          <p className="text-[8px] text-muted-foreground/60">
            &#10003; = matches the default threshold cascade leader
          </p>
        </div>
      )}

      {/* Note */}
      <p className="text-[9px] text-muted-foreground italic leading-relaxed">
        Cascade Map visualizes the temporal sequence of synchrony onset across modalities.
        This is a necessary but not sufficient condition for causal influence — Granger
        causality and cross-correlation lead-lag provide complementary directional evidence.
        Based on Koul et al. (2023) showing behavioral synchrony precedes neural synchrony onset.
      </p>
    </Card>
  );
};

export default CascadeMapPanel;
