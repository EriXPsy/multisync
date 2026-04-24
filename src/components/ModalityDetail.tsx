import { useParams } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  SYNCHRONY_INDICES,
  DEFAULT_WCC_PARAMS,
  type ModalityType,
  type SynchronyIndex,
} from "@/lib/synchrony-data";
import { Brain, Eye, Heart, Users, Settings2, BookOpen, Scale } from "lucide-react";

const MODALITY_META: Record<
  ModalityType,
  { label: string; icon: typeof Brain; colorVar: string; description: string; scoring: string; timescale: string }
> = {
  neural: {
    label: "Neural Synchrony",
    icon: Brain,
    colorVar: "--neural",
    description:
      "Inter-brain synchrony measured via EEG hyperscanning or fNIRS. Captures shared neural oscillatory patterns reflecting joint attention, social cognition, and coordinated processing.",
    scoring:
      "Each neural index (Alpha IBS, Theta IBS, PFC fNIRS, TPJ fNIRS) is computed at its native resolution (4-100ms) using PLV or WTC. Values are z-scored across the session, then weighted and summed to produce a composite. Weights reflect meta-analytic effect sizes (Tomashin et al., 2024: r=0.32 neural-behavioral association).",
    timescale:
      "Native: 4ms (EEG) – 100ms (fNIRS). Windowed analysis at 2-10s. Epoch-aggregated via mean of absolute WCC peaks within each common epoch.",
  },
  behavioral: {
    label: "Behavioral Synchrony",
    icon: Eye,
    colorVar: "--behavioral",
    description:
      "Temporal coordination of observable behaviors: gaze, head movements, gestures, facial expressions, and posture. Assessed via computer vision, motion capture, or manual coding.",
    scoring:
      "Each behavioral sub-index is computed independently via Windowed Cross-Correlation (WCC) at its native frame rate (~30Hz). The peak WCC correlation per window is extracted. Sub-indices are NOT simply averaged — they are z-scored individually, then combined via weighted sum where weights reflect the index's reliability and theoretical importance (Ramseyer & Tschacher, 2011). The composite is: Σ(wᵢ × zᵢ) / Σwᵢ.",
    timescale:
      "Native: 33ms (30Hz video). WCC windows: 5-10s. Each sub-index computes synchrony at its natural timescale, then epoch-aggregates to the common resolution by averaging peak-WCC values within each epoch.",
  },
  bio: {
    label: "Biosynchrony",
    icon: Heart,
    colorVar: "--bio",
    description:
      "Physiological co-regulation: alignment of heart rate (IBI), electrodermal activity (EDA), respiration, and pupil dilation between interaction partners.",
    scoring:
      "Each bio-index uses WCC with signal-specific parameters (Behrens et al., 2020 recommendations): HR uses 15s windows with ±5s lag, EDA uses 30s windows with ±7s lag, pupil uses 8s windows with ±3s lag. Peak absolute WCC per window is z-scored. Composite: weighted sum normalized by total weight. Surrogate testing (pseudo-pair shuffling) validates significance.",
    timescale:
      "Native: 33ms (pupil) – 1000ms (HR). WCC windows differ per signal (8-30s). Epoch aggregation: mean of peak-WCC values per common epoch. Signal-appropriate detrending applied before WCC.",
  },
  psycho: {
    label: "Psycho-synchrony",
    icon: Users,
    colorVar: "--psycho",
    description:
      "Subjective psychological alignment: perceived closeness (IOS), rapport, and shared flow states. Captured via discrete self-report at regular intervals during the interaction.",
    scoring:
      "IOS rated on 1-11 scale (IOS11, Baader et al., 2024) at configurable intervals (default: every 2 minutes). Rapport and flow on 7-point Likert scales. Dyadic convergence computed as 1 − |ratingA − ratingB| / scale_max. Z-scored across the session. Because ratings are sparse, interpolation is NOT applied — values are held constant within each rating epoch.",
    timescale:
      "Native: 2-3 min rating intervals. No windowed cross-correlation — instead, dyadic convergence per rating epoch. These discrete values are mapped to the common timeline as step-functions, not interpolated curves.",
  },
};

export function ModalityDetail() {
  const { modality } = useParams<{ modality: string }>();
  const mod = modality as ModalityType;
  const meta = MODALITY_META[mod];

  if (!meta) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-muted-foreground">Unknown modality: {modality}</p>
      </div>
    );
  }

  const indices = SYNCHRONY_INDICES.filter((i) => i.modality === mod);
  const Icon = meta.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: `hsl(var(${meta.colorVar}) / 0.15)` }}
        >
          <Icon className="w-5 h-5" style={{ color: `hsl(var(${meta.colorVar}))` }} />
        </div>
        <div>
          <h2 className="font-heading text-2xl font-bold">{meta.label}</h2>
          <p className="text-sm text-muted-foreground">{indices.length} sub-indices</p>
        </div>
      </div>

      {/* Description */}
      <Card className="glass-panel p-5 space-y-4">
        <p className="text-sm text-foreground/80 leading-relaxed">{meta.description}</p>
      </Card>

      {/* Methodology Accordion */}
      <Accordion type="multiple" defaultValue={["scoring", "timescale"]} className="space-y-2">
        <AccordionItem value="scoring" className="glass-panel rounded-lg border px-4">
          <AccordionTrigger className="hover:no-underline">
            <div className="flex items-center gap-2">
              <Scale className="w-4 h-4 text-accent" />
              <span className="font-heading text-sm font-semibold">Scoring & Composite Method</span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <p className="text-sm text-muted-foreground leading-relaxed">{meta.scoring}</p>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="timescale" className="glass-panel rounded-lg border px-4">
          <AccordionTrigger className="hover:no-underline">
            <div className="flex items-center gap-2">
              <Settings2 className="w-4 h-4 text-accent" />
              <span className="font-heading text-sm font-semibold">Timescale & Epoch Aggregation</span>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <p className="text-sm text-muted-foreground leading-relaxed">{meta.timescale}</p>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {/* Sub-indices Table */}
      <Card className="glass-panel overflow-hidden">
        <div className="p-4 border-b">
          <h3 className="font-heading text-sm font-semibold flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-accent" />
            Sub-indices & Weights
          </h3>
          <p className="text-[10px] text-muted-foreground mt-1">
            Composite = Σ(wᵢ × zᵢ) / Σwᵢ — each index z-scored at native resolution then weighted
          </p>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="font-heading text-xs">Index</TableHead>
              <TableHead className="font-heading text-xs">Weight</TableHead>
              <TableHead className="font-heading text-xs">Native Res.</TableHead>
              <TableHead className="font-heading text-xs">Method</TableHead>
              <TableHead className="font-heading text-xs">Reference</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {indices.map((idx) => {
              const wcc = DEFAULT_WCC_PARAMS[idx.id];
              return (
                <TableRow key={idx.id}>
                  <TableCell>
                    <div>
                      <p className="font-medium text-xs">{idx.name}</p>
                      <p className="text-[10px] text-muted-foreground max-w-[200px] truncate">
                        {idx.description}
                      </p>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2 min-w-[100px]">
                      <Progress value={idx.weight * 100} className="h-1.5 flex-1" />
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {idx.weight.toFixed(2)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="text-[10px] font-mono">
                      {idx.nativeResolutionMs}ms
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <p className="text-[10px] text-muted-foreground max-w-[180px]">
                      {idx.method}
                      {wcc && (
                        <span className="block text-[9px] mt-0.5 opacity-70">
                          Win: {wcc.windowSizeMs / 1000}s, Lag: ±{wcc.maxLagMs / 1000}s
                        </span>
                      )}
                    </p>
                  </TableCell>
                  <TableCell>
                    <p className="text-[10px] text-muted-foreground italic">
                      {idx.reference}
                    </p>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
}
