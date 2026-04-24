## SyncScope MVP - Three Features

### 1. Database Schema (Migration)
- **datasets** table: stores uploaded dataset metadata (name, description, modalities, created_at)
- **data_streams** table: stores imported CSV timeseries data linked to datasets (modality, index_type, sample_rate, data as JSONB)
- **video_annotations** table: stores behavioral event annotations from video coding (dataset_id, timestamp_ms, event_type, label, duration_ms)
- **analysis_runs** table: stores pipeline configuration and results (dataset_id, config JSONB, status, results JSONB)

### 2. Data Import Portal (`/import` page)
- Upload CSV/TSV files for each modality (neural, behavioral, bio, psycho)
- Parse CSV client-side, preview columns, map to synchrony indices
- Store parsed data in Supabase
- Support multiple streams per dataset

### 3. Video Annotation Tool (`/annotate` page)  
- Video player with timeline scrubber
- Predefined behavioral event buttons (gaze coordination, head movement, gesture, facial expression, etc.)
- Click button → record event at current video timestamp
- Event list with timestamps, editable/deletable
- Export annotations as data stream for pipeline

### 4. Analysis Pipeline (`/pipeline` page)
- Step 1: Select dataset
- Step 2: Choose data streams to include
- Step 3: Configure WCC parameters per stream
- Step 4: Set alignment parameters (common epoch, normalization)
- Step 5: Run analysis → show results on unified timeline
- Alignment verification: show per-stream native vs aligned timeseries overlay

### 5. Navigation
- Add new sidebar items: Import, Annotate, Pipeline
