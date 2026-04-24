import { useState, useRef, useCallback, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { supabase } from "@/integrations/supabase/client";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Video,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Eye,
  Hand,
  Smile,
  Move,
  MessageSquare,
  Trash2,
  Download,
  BookOpen,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  CheckCircle2,
  Info,
} from "lucide-react";
import { toast } from "sonner";
import { BEHAVIORAL_CODING_GUIDES, type BehavioralCodingGuide } from "@/lib/behavioral-coding-guide";

const EVENT_ICONS: Record<string, any> = {
  gaze_coordination: Eye,
  head_movement: Move,
  gesture_mirroring: Hand,
  facial_expression: Smile,
  vocal_sync: MessageSquare,
};

const EVENT_COLORS: Record<string, string> = {
  gaze_coordination: "hsl(185, 55%, 40%)",
  head_movement: "hsl(200, 55%, 45%)",
  gesture_mirroring: "hsl(160, 50%, 40%)",
  facial_expression: "hsl(35, 80%, 55%)",
  vocal_sync: "hsl(262, 60%, 55%)",
};

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = Math.floor(ms % 1000);
  return `${minutes}:${seconds.toString().padStart(2, "0")}.${milliseconds.toString().padStart(3, "0")}`;
}

const AnnotatePage = () => {
  const queryClient = useQueryClient();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedLevel, setSelectedLevel] = useState<Record<string, number>>({});
  const [expandedGuide, setExpandedGuide] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<
    { id?: string; timestamp_ms: number; event_type: string; label: string; level?: number; saved: boolean }[]
  >([]);

  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("datasets")
        .select("*")
        .order("created_at", { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  useEffect(() => {
    if (!selectedDataset) return;
    supabase
      .from("video_annotations")
      .select("*")
      .eq("dataset_id", selectedDataset)
      .order("timestamp_ms")
      .then(({ data }) => {
        if (data) {
          setAnnotations(
            data.map((a) => ({
              id: a.id,
              timestamp_ms: a.timestamp_ms,
              event_type: a.event_type,
              label: a.label,
              level: a.confidence ?? undefined,
              saved: true,
            }))
          );
        }
      });
  }, [selectedDataset]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === "Space") { e.preventDefault(); togglePlay(); }
      const guides = BEHAVIORAL_CODING_GUIDES;
      const num = parseInt(e.key);
      if (num >= 1 && num <= guides.length) {
        const guide = guides[num - 1];
        const level = selectedLevel[guide.type] ?? 2;
        markEvent(guide.type, guide.label, level);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [currentTime, selectedDataset, selectedLevel]);

  const handleVideoFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) videoRef.current.pause();
    else videoRef.current.play();
    setIsPlaying(!isPlaying);
  };

  const seekRelative = (ms: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime + ms / 1000);
  };

  const markEvent = useCallback(
    async (eventType: string, label: string, level?: number) => {
      const timestampMs = Math.round(currentTime * 1000);
      const newAnnotation = { timestamp_ms: timestampMs, event_type: eventType, label, level, saved: false };
      setAnnotations((prev) => [...prev, newAnnotation].sort((a, b) => a.timestamp_ms - b.timestamp_ms));

      if (selectedDataset) {
        const { data, error } = await supabase
          .from("video_annotations")
          .insert({
            dataset_id: selectedDataset,
            timestamp_ms: timestampMs,
            event_type: eventType,
            label,
            modality: "behavioral",
            confidence: level ?? null,
          })
          .select()
          .single();

        if (error) {
          toast.error("Failed to save annotation");
        } else {
          setAnnotations((prev) =>
            prev.map((a) =>
              a.timestamp_ms === timestampMs && a.event_type === eventType && !a.saved
                ? { ...a, id: data.id, saved: true }
                : a
            )
          );
          toast.success(`Marked ${label} (L${level ?? "?"}) at ${formatTimestamp(timestampMs)}`);
        }
      } else {
        toast.info("Select a dataset to persist annotations");
      }
    },
    [currentTime, selectedDataset]
  );

  const deleteAnnotation = async (idx: number) => {
    const ann = annotations[idx];
    if (ann.id && selectedDataset) {
      await supabase.from("video_annotations").delete().eq("id", ann.id);
    }
    setAnnotations((prev) => prev.filter((_, i) => i !== idx));
  };

  const exportAnnotations = () => {
    const csv = [
      "timestamp_ms,event_type,label,level",
      ...annotations.map((a) => `${a.timestamp_ms},${a.event_type},${a.label},${a.level ?? ""}`),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `annotations_${selectedDataset || "unsaved"}.csv`;
    link.click();
  };

  return (
    <div className="p-6 max-w-[1200px] space-y-6">
      <div>
        <h2 className="font-heading text-2xl font-bold">Video Annotation Tool</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Code behavioral synchrony events with structured coding guides
        </p>
      </div>

      {/* Dataset & Video */}
      <Card className="glass-panel p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-xs font-heading">Link to Dataset</Label>
            <Select value={selectedDataset} onValueChange={setSelectedDataset}>
              <SelectTrigger className="h-9 text-sm"><SelectValue placeholder="Select dataset..." /></SelectTrigger>
              <SelectContent>
                {datasets?.map((ds) => (
                  <SelectItem key={ds.id} value={ds.id}>{ds.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-heading">Load Video File</Label>
            <label>
              <input type="file" accept="video/*" onChange={handleVideoFile} className="hidden" />
              <Button variant="outline" size="sm" asChild>
                <span className="cursor-pointer">
                  <Video className="w-4 h-4 mr-2" />
                  {videoUrl ? "Change Video" : "Select Video"}
                </span>
              </Button>
            </label>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Player */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="glass-panel overflow-hidden">
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                className="w-full aspect-video bg-black"
                onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
                onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
              />
            ) : (
              <div className="w-full aspect-video bg-muted/50 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                  <Video className="w-12 h-12 mx-auto mb-2 opacity-30" />
                  <p className="text-sm">Load a video to begin annotation</p>
                </div>
              </div>
            )}

            <div className="p-3 border-t border-border">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => seekRelative(-5000)}>
                    <SkipBack className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-8 w-8" onClick={togglePlay} disabled={!videoUrl}>
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>
                  <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => seekRelative(5000)}>
                    <SkipForward className="w-4 h-4" />
                  </Button>
                </div>
                <div className="font-mono text-xs text-muted-foreground">
                  {formatTimestamp(currentTime * 1000)} / {formatTimestamp(duration * 1000)}
                </div>
              </div>
              {videoUrl && (
                <input
                  type="range"
                  min={0}
                  max={duration || 1}
                  step={0.01}
                  value={currentTime}
                  onChange={(e) => {
                    if (videoRef.current) videoRef.current.currentTime = parseFloat(e.target.value);
                  }}
                  className="w-full mt-2 h-1 accent-primary"
                />
              )}
            </div>
          </Card>

          {/* Event Coding with Guides */}
          <Card className="glass-panel p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-heading text-sm font-semibold">
                Mark Event at {formatTimestamp(currentTime * 1000)}
              </h3>
              <Badge variant="outline" className="text-[10px]">
                <Info className="w-3 h-3 mr-1" /> Keys 1-5 for quick mark, Space for play/pause
              </Badge>
            </div>

            <div className="space-y-2">
              {BEHAVIORAL_CODING_GUIDES.map((guide) => {
                const Icon = EVENT_ICONS[guide.type] || Eye;
                const color = EVENT_COLORS[guide.type] || "hsl(var(--accent))";
                const level = selectedLevel[guide.type] ?? 2;
                const isExpanded = expandedGuide === guide.type;

                return (
                  <div key={guide.type} className="rounded-lg border border-border/50 overflow-hidden">
                    {/* Event Button Row */}
                    <div className="flex items-center gap-2 p-2 bg-muted/20">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => markEvent(guide.type, guide.label, level)}
                        disabled={!videoUrl}
                        className="gap-1.5 min-w-[160px] justify-start"
                      >
                        <Icon className="w-4 h-4 flex-shrink-0" style={{ color }} />
                        <span className="text-[11px]">{guide.label}</span>
                        <Badge variant="secondary" className="ml-auto text-[9px] h-4 px-1">{guide.keyboardShortcut}</Badge>
                      </Button>

                      {/* Level Selector */}
                      <div className="flex items-center gap-1">
                        {guide.levels.map((l) => (
                          <button
                            key={l.level}
                            onClick={() => setSelectedLevel((prev) => ({ ...prev, [guide.type]: l.level }))}
                            className={`w-7 h-7 rounded-md text-[10px] font-bold transition-all ${
                              level === l.level
                                ? "ring-2 ring-accent bg-accent/20 text-foreground"
                                : "bg-muted/50 text-muted-foreground hover:bg-muted"
                            }`}
                            title={`${l.label}: ${l.description}`}
                          >
                            {l.level}
                          </button>
                        ))}
                      </div>

                      {/* Guide Toggle */}
                      <Collapsible open={isExpanded} onOpenChange={() => setExpandedGuide(isExpanded ? null : guide.type)}>
                        <CollapsibleTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-7 w-7 ml-auto">
                            <BookOpen className="w-3.5 h-3.5" />
                          </Button>
                        </CollapsibleTrigger>
                      </Collapsible>
                    </div>

                    {/* Expanded Coding Guide */}
                    {isExpanded && (
                      <CodingGuidePanel guide={guide} />
                    )}
                  </div>
                );
              })}
            </div>
          </Card>
        </div>

        {/* Annotation List */}
        <div className="space-y-4">
          <Card className="glass-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-heading text-sm font-semibold">
                Annotations ({annotations.length})
              </h3>
              {annotations.length > 0 && (
                <Button variant="ghost" size="sm" onClick={exportAnnotations} className="h-7 text-[10px]">
                  <Download className="w-3 h-3 mr-1" />
                  Export CSV
                </Button>
              )}
            </div>

            {annotations.length === 0 ? (
              <p className="text-xs text-muted-foreground">No events annotated yet.</p>
            ) : (
              <div className="space-y-1 max-h-[500px] overflow-y-auto">
                {annotations.map((ann, i) => {
                  const color = EVENT_COLORS[ann.event_type] || "hsl(var(--accent))";
                  return (
                    <div
                      key={i}
                      className="flex items-center justify-between p-2 rounded-md bg-muted/30 hover:bg-muted/50 cursor-pointer group"
                      onClick={() => {
                        if (videoRef.current) videoRef.current.currentTime = ann.timestamp_ms / 1000;
                      }}
                    >
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                        <div>
                          <div className="flex items-center gap-1">
                            <p className="text-[11px] font-medium">{ann.label}</p>
                            {ann.level !== undefined && (
                              <Badge variant="outline" className="text-[8px] h-4 px-1">L{ann.level}</Badge>
                            )}
                          </div>
                          <p className="text-[10px] text-muted-foreground font-mono">
                            {formatTimestamp(ann.timestamp_ms)}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        {ann.saved && (
                          <Badge variant="outline" className="text-[8px] h-4">saved</Badge>
                        )}
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 opacity-0 group-hover:opacity-100"
                          onClick={(e) => { e.stopPropagation(); deleteAnnotation(i); }}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

function CodingGuidePanel({ guide }: { guide: BehavioralCodingGuide }) {
  return (
    <div className="p-3 bg-muted/10 border-t border-border/30 space-y-3 text-xs">
      <div>
        <p className="font-heading font-semibold text-[11px] mb-1">Definition</p>
        <p className="text-muted-foreground leading-relaxed">{guide.definition}</p>
      </div>

      <div>
        <p className="font-heading font-semibold text-[11px] mb-1 flex items-center gap-1">
          <CheckCircle2 className="w-3 h-3 text-accent" /> What to Look For
        </p>
        <ul className="space-y-0.5 text-muted-foreground">
          {guide.whatToLookFor.map((item, i) => (
            <li key={i} className="flex items-start gap-1.5">
              <span className="text-accent mt-0.5">•</span>
              <span className="leading-relaxed">{item}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <p className="font-heading font-semibold text-[11px] mb-1 flex items-center gap-1">
          <AlertTriangle className="w-3 h-3 text-psycho" /> Common Mistakes
        </p>
        <ul className="space-y-0.5 text-muted-foreground">
          {guide.commonMistakes.map((item, i) => (
            <li key={i} className="flex items-start gap-1.5">
              <span className="text-psycho mt-0.5">•</span>
              <span className="leading-relaxed">{item}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <p className="font-heading font-semibold text-[11px] mb-1">Levels</p>
        <div className="grid grid-cols-2 gap-2">
          {guide.levels.map((level) => (
            <div key={level.level} className="rounded-md bg-muted/30 p-2 space-y-1">
              <div className="flex items-center gap-1.5">
                <span
                  className="w-5 h-5 rounded text-[10px] font-bold flex items-center justify-center text-background"
                  style={{ backgroundColor: level.color }}
                >
                  {level.level}
                </span>
                <span className="font-semibold text-[10px]">{level.label}</span>
              </div>
              <p className="text-[9px] text-muted-foreground">{level.description}</p>
              <div className="text-[9px] text-muted-foreground/70 italic">
                e.g., {level.examples.join("; ")}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="text-[9px] text-muted-foreground/60">
        Refs: {guide.references.join("; ")}
      </div>
    </div>
  );
}

export default AnnotatePage;
