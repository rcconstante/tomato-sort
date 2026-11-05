export type Decision = "ACCEPT" | "REJECT" | "REVIEW";
export type PipelineState = "idle" | "running" | "paused" | "finished" | "stopped" | "error";
export type SourceType = "upload" | "file" | "webcam" | "rtsp";

export interface InspectionLine {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  direction: "forward" | "reverse" | "both";
}

export interface Settings {
  detect_conf: number;
  certain_conf: number;
  min_observations: number;
  grading_mode: "binary" | "strict";
  count_trigger: "detection" | "line";
  min_vote_share: number;
  imgsz: number;
  line: InspectionLine;
  show_boxes: boolean;
  outline_mode: "contour" | "box";
  show_line: boolean;
  loop_video: boolean;
  realtime_playback: boolean;
  sorting: Record<string, Decision>;
}

export interface SourceInfo {
  type?: SourceType;
  name?: string;
  live?: boolean;
  width?: number;
  height?: number;
  fps?: number | null;
  frames?: number | null;
}

export interface PipelineStatus {
  state: PipelineState;
  error: string | null;
  session_id: number | null;
  source: SourceInfo;
  frame_index: number;
  loops: number;
  fps: number;
  inference_ms: number;
  active_tracks: number;
  throughput_per_min: number;
  counts: { total: number; labels: Record<string, number>; decisions: Record<Decision, number> };
  last_event_seq: number;
}

export interface InspectionEvent {
  id: number;
  seq: number;
  number: number;
  session_id: number;
  track_id: number;
  label: string;
  category: string;
  confidence: number;
  predicted_label: string | null;
  vote_share: number;
  observations: number;
  decision: Decision;
  reason: string;
  frame_index: number;
  created_at: string;
  votes: Record<string, number>;
  crop: string | null;
}

export interface ModelStatus {
  installed: boolean;
  path: string;
  error: string | null;
  classes: Record<string, string>;
  device: string;
  metadata: {
    model_architecture?: string;
    dataset?: { project?: string; version?: number };
    imgsz?: number;
    training_date?: string;
    metrics?: { val?: Record<string, number> | null; test?: Record<string, number> | null };
  } | null;
}
