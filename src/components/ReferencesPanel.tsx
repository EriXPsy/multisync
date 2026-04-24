import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { REFERENCES } from "@/lib/synchrony-data";
import { BookOpen } from "lucide-react";

const CATEGORY_LABELS: Record<string, string> = {
  theory: "Theory",
  methods: "Methods",
  neural: "Neural",
  behavioral: "Behavioral",
  bio: "Physiological",
  psycho: "Psycho",
  multimodal: "Multimodal",
};

const CATEGORY_COLORS: Record<string, string> = {
  theory: "bg-accent/10 text-accent",
  methods: "bg-primary/10 text-primary",
  neural: "bg-neural/10 text-neural",
  behavioral: "bg-behavioral/10 text-behavioral",
  bio: "bg-bio/10 text-bio",
  psycho: "bg-psycho/10 text-psycho",
  multimodal: "bg-highlight/10 text-highlight",
};

export function ReferencesPanel() {
  const categories = [...new Set(REFERENCES.map((r) => r.category))];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-heading text-2xl font-bold">Literature References</h2>
        <p className="text-sm text-muted-foreground mt-1">
          {REFERENCES.length} studies grounding the multimodal synchrony framework
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {categories.map((cat) => (
          <Badge key={cat} variant="outline" className="text-xs">
            {CATEGORY_LABELS[cat] || cat}: {REFERENCES.filter((r) => r.category === cat).length}
          </Badge>
        ))}
      </div>

      <div className="space-y-3">
        {REFERENCES.map((ref) => (
          <Card key={ref.key} className="glass-panel p-4 flex items-start gap-3">
            <BookOpen className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
            <div className="space-y-1 flex-1 min-w-0">
              <p className="text-xs text-foreground/90 leading-relaxed">{ref.citation}</p>
              <Badge className={`text-[10px] border-none ${CATEGORY_COLORS[ref.category] || ""}`}>
                {CATEGORY_LABELS[ref.category] || ref.category}
              </Badge>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
