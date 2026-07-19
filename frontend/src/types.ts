/** Single detected object */
export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] normalized 0-1
  tracker_id: number | null;
  tracking_duration_frames?: number;
  zone_id?: number | null;
}

/** Frame detection data from backend */
export interface DetectionData {
  type: 'detection';
  detections: Detection[];
  session_counts: Record<string, number>;
  total_objects: number;
  active_tracks_count: number;
  total_unique_seen: number;
  fps: number;
  inference_ms: number;
  frame_width: number;
  frame_height: number;
  model?: string;
}

/** Detection statistics */
export interface DetectorStats {
  fps: number;
  frames_processed: number;
  uptime_seconds: number;
  session_counts: Record<string, number>;
  conf_threshold: number;
  iou_threshold: number;
  device: string;
  tracker: string;
  total_tracks_created: number;
  total_id_switches: number;
  tracking_stability: number;
  model: string;
  model_info: ModelInfo;
  imgsz: number;
  class_subset: string;
  active_classes: number;
}

/** Model information */
export interface ModelInfo {
  file: string;
  params: string;
  map: number;
  speed_ms: number;
}

/** Available models */
export interface ModelsResponse {
  current: string | null;
  available: Record<string, ModelInfo>;
  device: string;
}

/** Congestion level */
export type CongestionLevel = 'low' | 'medium' | 'high' | 'severe';
