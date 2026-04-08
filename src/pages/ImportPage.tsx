import { useState, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { supabase } from "@/integrations/supabase/client";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Upload, FileSpreadsheet, Plus, Trash2, Check, AlertCircle, Database } from "lucide-react";
import { toast } from "sonner";
import { SYNCHRONY_INDICES } from "@/lib/synchrony-data";

type ParsedCSV = {
  headers: string[];
  rows: number[][];
  preview: number[][];
};

function parseCSV(text: string): ParsedCSV {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(/[,\t]/).map((h) => h.trim().replace(/^"|"$/g, ""));
  const rows = lines.slice(1).map((line) =>
    line.split(/[,\t]/).map((v) => {
      const n = parseFloat(v.trim().replace(/^"|"$/g, ""));
      return isNaN(n) ? 0 : n;
    })
  );
  return { headers, rows, preview: rows.slice(0, 5) };
}

type StreamMapping = {
  file: File;
  parsed: ParsedCSV;
  modality: string;
  indexName: string;
  timestampCol: number;
  person1Col: number;
  person2Col: number;
  sampleRateHz: number;
  unit: string;
};

const ImportPage = () => {
  const queryClient = useQueryClient();
  const [datasetName, setDatasetName] = useState("");
  const [datasetDesc, setDatasetDesc] = useState("");
  const [streams, setStreams] = useState<StreamMapping[]>([]);
  const [uploading, setUploading] = useState(false);

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

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    for (const file of Array.from(files)) {
      const text = await file.text();
      const parsed = parseCSV(text);
      setStreams((prev) => [
        ...prev,
        {
          file,
          parsed,
          modality: "behavioral",
          indexName: "",
          timestampCol: 0,
          person1Col: 1,
          person2Col: 2,
          sampleRateHz: 30,
          unit: "a.u.",
        },
      ]);
    }
    e.target.value = "";
  }, []);

  const updateStream = (idx: number, updates: Partial<StreamMapping>) => {
    setStreams((prev) => prev.map((s, i) => (i === idx ? { ...s, ...updates } : s)));
  };

  const removeStream = (idx: number) => {
    setStreams((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleImport = async () => {
    if (!datasetName.trim()) {
      toast.error("Please enter a dataset name");
      return;
    }
    if (streams.length === 0) {
      toast.error("Please add at least one data stream");
      return;
    }

    setUploading(true);
    try {
      const modalities = [...new Set(streams.map((s) => s.modality))];
      const { data: dataset, error: dsError } = await supabase
        .from("datasets")
        .insert({
          name: datasetName,
          description: datasetDesc,
          modalities,
          status: "processing",
        })
        .select()
        .single();

      if (dsError) throw dsError;

      for (const stream of streams) {
        const timeseriesData = stream.parsed.rows.map((row) => ({
          t: row[stream.timestampCol] ?? 0,
          p1: row[stream.person1Col] ?? 0,
          p2: row[stream.person2Col] ?? 0,
        }));

        const { error } = await supabase.from("data_streams").insert({
          dataset_id: dataset.id,
          modality: stream.modality,
          index_name: stream.indexName || stream.file.name.replace(/\.\w+$/, ""),
          sample_rate_hz: stream.sampleRateHz,
          unit: stream.unit,
          column_mapping: {
            timestamp: stream.parsed.headers[stream.timestampCol],
            person1: stream.parsed.headers[stream.person1Col],
            person2: stream.parsed.headers[stream.person2Col],
          },
          data: timeseriesData,
        });
        if (error) throw error;
      }

      await supabase.from("datasets").update({ status: "complete" }).eq("id", dataset.id);
      toast.success(`Dataset "${datasetName}" imported with ${streams.length} streams`);
      setDatasetName("");
      setDatasetDesc("");
      setStreams([]);
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    } catch (err: any) {
      toast.error(err.message || "Import failed");
    } finally {
      setUploading(false);
    }
  };

  const indexOptions = SYNCHRONY_INDICES.map((idx) => ({
    value: idx.id,
    label: `${idx.name} (${idx.modality})`,
    modality: idx.modality,
  }));

  return (
    <div className="p-6 max-w-[1000px] space-y-6">
      <div>
        <h2 className="font-heading text-2xl font-bold">Data Import Portal</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Import CSV/TSV timeseries data for multimodal synchrony analysis
        </p>
      </div>

      <Card className="glass-panel p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Plus className="w-4 h-4 text-accent" />
          <h3 className="font-heading text-sm font-semibold">New Dataset</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-xs font-heading">Dataset Name</Label>
            <Input
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="e.g., Dyad_001_Session_1"
              className="h-9 text-sm"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-xs font-heading">Description</Label>
            <Input
              value={datasetDesc}
              onChange={(e) => setDatasetDesc(e.target.value)}
              placeholder="Optional description"
              className="h-9 text-sm"
            />
          </div>
        </div>

        <Separator />

        <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
          <Upload className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
          <p className="text-sm text-muted-foreground mb-2">
            Drop CSV/TSV files here, or click to browse
          </p>
          <p className="text-[10px] text-muted-foreground mb-3">
            Each file = one synchrony index. Columns: timestamp, person1_signal, person2_signal
          </p>
          <label>
            <input
              type="file"
              accept=".csv,.tsv,.txt"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
            <Button variant="outline" size="sm" asChild>
              <span className="cursor-pointer">
                <FileSpreadsheet className="w-4 h-4 mr-2" />
                Select Files
              </span>
            </Button>
          </label>
        </div>

        {streams.map((stream, idx) => (
          <Card key={idx} className="p-4 bg-muted/30 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="w-4 h-4 text-accent" />
                <span className="text-sm font-medium font-heading">{stream.file.name}</span>
                <Badge variant="secondary" className="text-[10px]">
                  {stream.parsed.headers.length} cols × {stream.parsed.rows.length} rows
                </Badge>
              </div>
              <Button variant="ghost" size="icon" onClick={() => removeStream(idx)} className="h-7 w-7">
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="space-y-1">
                <Label className="text-[10px]">Modality</Label>
                <Select value={stream.modality} onValueChange={(v) => updateStream(idx, { modality: v })}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="neural">Neural</SelectItem>
                    <SelectItem value="behavioral">Behavioral</SelectItem>
                    <SelectItem value="bio">Biosynchrony</SelectItem>
                    <SelectItem value="psycho">Psycho-synchrony</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">Index Name</Label>
                <Input
                  value={stream.indexName}
                  onChange={(e) => updateStream(idx, { indexName: e.target.value })}
                  placeholder="e.g., gaze_coordination"
                  className="h-8 text-xs"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">Sample Rate (Hz)</Label>
                <Input
                  type="number"
                  value={stream.sampleRateHz}
                  onChange={(e) => updateStream(idx, { sampleRateHz: parseFloat(e.target.value) || 0 })}
                  className="h-8 text-xs"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-[10px]">Unit</Label>
                <Input
                  value={stream.unit}
                  onChange={(e) => updateStream(idx, { unit: e.target.value })}
                  className="h-8 text-xs"
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Timestamp Column", key: "timestampCol" as const },
                { label: "Person 1 Signal", key: "person1Col" as const },
                { label: "Person 2 Signal", key: "person2Col" as const },
              ].map((col) => (
                <div key={col.key} className="space-y-1">
                  <Label className="text-[10px]">{col.label}</Label>
                  <Select
                    value={String(stream[col.key])}
                    onValueChange={(v) => updateStream(idx, { [col.key]: parseInt(v) })}
                  >
                    <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {stream.parsed.headers.map((h, hi) => (
                        <SelectItem key={hi} value={String(hi)}>{h}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ))}
            </div>

            <div className="overflow-x-auto">
              <table className="text-[10px] w-full">
                <thead>
                  <tr>
                    {stream.parsed.headers.map((h, i) => (
                      <th key={i} className="px-2 py-1 text-left text-muted-foreground font-medium">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {stream.parsed.preview.map((row, ri) => (
                    <tr key={ri} className="border-t border-border/30">
                      {row.map((val, ci) => (
                        <td key={ci} className="px-2 py-1 font-mono">{val.toFixed(4)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        ))}

        {streams.length > 0 && (
          <Button onClick={handleImport} disabled={uploading} className="w-full">
            {uploading ? (
              "Importing..."
            ) : (
              <>
                <Database className="w-4 h-4 mr-2" />
                Import {streams.length} Stream{streams.length > 1 ? "s" : ""} to Database
              </>
            )}
          </Button>
        )}
      </Card>

      <Card className="glass-panel p-5 space-y-3">
        <h3 className="font-heading text-sm font-semibold">Existing Datasets</h3>
        {!datasets || datasets.length === 0 ? (
          <p className="text-xs text-muted-foreground">No datasets imported yet.</p>
        ) : (
          <div className="space-y-2">
            {datasets.map((ds) => (
              <div key={ds.id} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                <div>
                  <p className="text-sm font-medium font-heading">{ds.name}</p>
                  <p className="text-[10px] text-muted-foreground">{ds.description}</p>
                </div>
                <div className="flex items-center gap-2">
                  {ds.modalities?.map((m) => (
                    <Badge key={m} variant="outline" className="text-[10px] capitalize">{m}</Badge>
                  ))}
                  <Badge
                    variant={ds.status === "complete" ? "default" : "secondary"}
                    className="text-[10px]"
                  >
                    {ds.status === "complete" && <Check className="w-3 h-3 mr-1" />}
                    {ds.status === "error" && <AlertCircle className="w-3 h-3 mr-1" />}
                    {ds.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

export default ImportPage;
