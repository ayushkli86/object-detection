/** Single detected object */
export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] normalized 0-1
  tracker_id: number | null; // Persistent unique ID from ByteTrack
}

/** Frame detection data from backend */
export interface DetectionData {
  type: 'detection';
  capture_mode: 'screen' | 'camera';
  detections: Detection[];
  object_counts: Record<string, number>;
  session_counts: Record<string, number>;
  total_objects: number;
  active_tracks_count: number;
  total_unique_seen: number;
  fps: number;
  inference_ms: number;
  capture_ms: number;
  frame_width: number;
  frame_height: number;
  image?: string; // base64 data URL of annotated frame (camera mode)
}

/** Detection statistics */
export interface DetectorStats {
  fps: number;
  frames_processed: number;
  uptime_seconds: number;
  session_counts: Record<string, number>;
  cumulative_counts: Record<string, number>;
  total_objects_detected: number;
  unique_classes_detected: number;
  conf_threshold: number;
  iou_threshold: number;
  device: string;
}
