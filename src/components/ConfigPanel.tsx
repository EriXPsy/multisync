import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { DEFAULT_WCC_PARAMS, SYNCHRONY_INDICES } from "@/lib/synchrony-data";
import { AlertTriangle, CheckCircle2, Lightbulb, Wrench } from "lucide-react";

export function ConfigPanel() {
  const modalities = ["neural", "behavioral", "bio", "psycho"] as const;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-heading text-2xl font-bold">Pipeline Configuration</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Multi-timescale alignment strategy and processing parameters
        </p>
      </div>

      {/* Architecture Overview */}
      <Card className="glass-panel p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Wrench className="w-4 h-4 text-accent" />
          <h3 className="font-heading text-sm font-semibold">Processing Architecture</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <h4 className="text-xs font-heading font-semibold text-muted-foreground uppercase tracking-wide">
              Per-Stream Processing
            </h4>
            <ol className="text-xs text-foreground/80 space-y-2 list-decimal list-inside">
              <li><strong>Acquire</strong> — Raw signal at native sampling rate</li>
              <li><strong>Preprocess</strong> — Band-pass filter, artifact rejection, detrending</li>
              <li><strong>Compute WCC</strong> — Using signal-specific window/lag params</li>
              <li><strong>Extract peak</strong> — Max absolute WCC per window</li>
              <li><strong>Z-score</strong> — Standardize across session within-index</li>
            </ol>
          </div>
          <div className="space-y-3">
            <h4 className="text-xs font-heading font-semibold text-muted-foreground uppercase tracking-wide">
              Cross-Modal Alignment
            </h4>
            <ol className="text-xs text-foreground/80 space-y-2 list-decimal list-inside">
              <li><strong>Weighted composite</strong> — Σ(wᵢ × zᵢ) / Σwᵢ per modality</li>
              <li><strong>Epoch aggregate</strong> — Mean of composite within common epoch</li>
              <li><strong>Unified timeline</strong> — All modalities at configurable resolution</li>
              <li><strong>Cascade detect</strong> — Onset threshold crossing per modality</li>
              <li><strong>Surrogate test</strong> — Pseudo-pair shuffling for significance</li>
            </ol>
          </div>
        </div>
      </Card>

      {/* Multi-timescale Problem */}
      <Card className="glass-panel p-5 space-y-4 border-l-4 border-l-psycho">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-psycho" />
          <h3 className="font-heading text-sm font-semibold">Multi-Timescale Challenge</h3>
        </div>
        <div className="text-xs text-foreground/80 space-y-3 leading-relaxed">
          <p>
            <strong>Problem:</strong> Neural data (EEG) are sampled at ~250Hz (4ms), behavioral indices at ~30Hz (33ms), 
            biosignals at 1-4Hz (250ms-1s), and psycho-synchrony ratings every 2-3 minutes. Direct comparison is invalid 
            without alignment.
          </p>
          <p>
            <strong>Solution:</strong> Each stream computes synchrony at its <em>native</em> window size and resolution 
            (per Behrens et al., 2020 recommendations). The resulting time-varying synchrony series is then 
            <em>epoch-aggregated</em> to a configurable common resolution by averaging synchrony values falling 
            within each epoch boundary. This preserves signal-specific dynamics while enabling cross-modal comparison.
          </p>
          <p>
            <strong>For psycho-synchrony:</strong> Discrete IOS/rapport ratings are not interpolated — they are held 
            constant as step-functions within their rating epoch, then mapped to the common timeline. This avoids 
            artificial smoothing of inherently discrete subjective data (Baader et al., 2024).
          </p>
        </div>
      </Card>

      {/* Key Design Decisions */}
      <Card className="glass-panel p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Lightbulb className="w-4 h-4 text-psycho" />
          <h3 className="font-heading text-sm font-semibold">Key Design Decisions</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[
            {
              q: "Why weighted composite, not simple average?",
              a: "Sub-indices vary in reliability, effect size, and theoretical importance. Meta-analytic evidence (Tomashin et al., 2024: neural-behavioral r=0.32, physio-behavioral r=0.18) suggests differential associations. Weights allow incorporating this evidence.",
            },
            {
              q: "Why z-score rather than min-max?",
              a: "Z-scoring preserves distributional properties and is robust to outliers. Min-max rescaling is sensitive to extreme values and loses information about within-session variability. Z-scores also enable parametric cascade detection via threshold crossing.",
            },
            {
              q: "Why not interpolate psycho-synchrony?",
              a: "IOS ratings are inherently discrete, sparse measurements. Linear or spline interpolation would create artificial 'micro-dynamics' that don't exist in the subjective experience. Step-function representation is methodologically honest.",
            },
            {
              q: "Why WCC over other methods?",
              a: "WCC (Boker et al., 2002) captures time-varying synchrony AND lead-lag dynamics, which is essential for cascade detection. Cross-recurrence quantification or wavelet coherence are alternatives for specific use cases (Meier & Tschacher, 2021).",
            },
          ].map((item, i) => (
            <div key={i} className="bg-muted/50 rounded-lg p-3 space-y-1">
              <p className="text-xs font-heading font-semibold">{item.q}</p>
              <p className="text-[10px] text-muted-foreground leading-relaxed">{item.a}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* WCC Parameters per Index */}
      <Accordion type="multiple" className="space-y-2">
        {modalities.map((mod) => {
          const indices = SYNCHRONY_INDICES.filter((i) => i.modality === mod);
          return (
            <AccordionItem key={mod} value={mod} className="glass-panel rounded-lg border px-4">
              <AccordionTrigger className="hover:no-underline">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-accent" />
                  <span className="font-heading text-sm font-semibold capitalize">{mod}</span>
                  <Badge variant="secondary" className="text-[10px]">{indices.length} indices</Badge>
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3">
                  {indices.map((idx) => {
                    const params = DEFAULT_WCC_PARAMS[idx.id];
                    return (
                      <div key={idx.id} className="bg-muted/30 rounded-md p-3 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium">{idx.name}</span>
                          <Badge variant="outline" className="text-[10px] font-mono">
                            {idx.unit}
                          </Badge>
                        </div>
                        {params && (
                          <div className="grid grid-cols-4 gap-2">
                            {[
                              { label: "Window", value: `${params.windowSizeMs / 1000}s` },
                              { label: "Max Lag", value: `±${params.maxLagMs / 1000}s` },
                              { label: "Win Inc.", value: `${params.windowIncrementMs}ms` },
                              { label: "Lag Inc.", value: `${params.lagIncrementMs}ms` },
                            ].map((p) => (
                              <div key={p.label} className="text-center">
                                <p className="text-[9px] text-muted-foreground">{p.label}</p>
                                <p className="text-[11px] font-mono font-medium">{p.value}</p>
                              </div>
                            ))}
                          </div>
                        )}
                        <Separator />
                        <p className="text-[10px] text-muted-foreground">{idx.method}</p>
                      </div>
                    );
                  })}
                </div>
              </AccordionContent>
            </AccordionItem>
          );
        })}
      </Accordion>
    </div>
  );
}
