
-- Create datasets table
CREATE TABLE public.datasets (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'processing', 'complete', 'error')),
  modalities TEXT[] DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read datasets" ON public.datasets FOR SELECT USING (true);
CREATE POLICY "Public insert datasets" ON public.datasets FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update datasets" ON public.datasets FOR UPDATE USING (true);
CREATE POLICY "Public delete datasets" ON public.datasets FOR DELETE USING (true);

-- Create data_streams table
CREATE TABLE public.data_streams (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  dataset_id UUID NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
  modality TEXT NOT NULL CHECK (modality IN ('neural', 'behavioral', 'bio', 'psycho')),
  index_name TEXT NOT NULL,
  sample_rate_hz NUMERIC,
  unit TEXT,
  column_mapping JSONB DEFAULT '{}',
  data JSONB DEFAULT '[]',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.data_streams ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read data_streams" ON public.data_streams FOR SELECT USING (true);
CREATE POLICY "Public insert data_streams" ON public.data_streams FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update data_streams" ON public.data_streams FOR UPDATE USING (true);
CREATE POLICY "Public delete data_streams" ON public.data_streams FOR DELETE USING (true);

CREATE INDEX idx_data_streams_dataset ON public.data_streams(dataset_id);

-- Create video_annotations table
CREATE TABLE public.video_annotations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  dataset_id UUID NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
  timestamp_ms NUMERIC NOT NULL,
  end_timestamp_ms NUMERIC,
  event_type TEXT NOT NULL,
  label TEXT NOT NULL,
  modality TEXT NOT NULL DEFAULT 'behavioral' CHECK (modality IN ('neural', 'behavioral', 'bio', 'psycho')),
  confidence NUMERIC DEFAULT 1.0,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.video_annotations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read video_annotations" ON public.video_annotations FOR SELECT USING (true);
CREATE POLICY "Public insert video_annotations" ON public.video_annotations FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update video_annotations" ON public.video_annotations FOR UPDATE USING (true);
CREATE POLICY "Public delete video_annotations" ON public.video_annotations FOR DELETE USING (true);

CREATE INDEX idx_video_annotations_dataset ON public.video_annotations(dataset_id);

-- Create analysis_runs table
CREATE TABLE public.analysis_runs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  dataset_id UUID NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
  name TEXT NOT NULL DEFAULT 'Untitled Analysis',
  config JSONB NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'complete', 'error')),
  results JSONB DEFAULT '{}',
  alignment_report JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.analysis_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read analysis_runs" ON public.analysis_runs FOR SELECT USING (true);
CREATE POLICY "Public insert analysis_runs" ON public.analysis_runs FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update analysis_runs" ON public.analysis_runs FOR UPDATE USING (true);
CREATE POLICY "Public delete analysis_runs" ON public.analysis_runs FOR DELETE USING (true);

CREATE INDEX idx_analysis_runs_dataset ON public.analysis_runs(dataset_id);

-- Timestamp update function
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = public;

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON public.datasets FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_analysis_runs_updated_at BEFORE UPDATE ON public.analysis_runs FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
