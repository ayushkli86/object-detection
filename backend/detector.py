"""
YOLOv8 Object Detector with ByteTrack Tracking
Uses Ultralytics YOLOv8 pretrained on COCO dataset (80 classes)
Each unique object gets a persistent tracker_id across frames.

Stability improvements:
  - Bbox Kalman smoothing to reduce jitter
  - ID switch detection for tracking stability metrics
  - Confidence smoothing across frames
  - Optimized ByteTrack config for better track persistence
"""

import os
import cv2
import numpy as np
import ctypes
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
import logging
import glob
import collections


# ---------------------------------------------------------------------------
# Auto-detect CUDA library path
# ---------------------------------------------------------------------------
def _setup_cuda_library_path():
    try:
        ctypes.CDLL('libcuda.so.1')
        return
    except OSError:
        pass

    search_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/opt/cuda/lib64',
        '/var/lib/flatpak/runtime/org.freedesktop.Platform.GL.nvidia-*/x86_64/*/*/files/lib',
        '/var/lib/flatpak/runtime/org.freedesktop.Platform.GL.nvidia-*/x86_64/*/*/lib',
    ]

    for pattern in search_paths:
        for lib_dir in glob.glob(pattern):
            for name in ('libcuda.so.1', 'libcuda.so'):
                lib_path = os.path.join(lib_dir, name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.cdll.LoadLibrary(lib_path)
                        logger.info(f"CUDA driver preloaded from: {lib_path}")
                        current = os.environ.get('LD_LIBRARY_PATH', '')
                        if lib_dir not in current:
                            os.environ['LD_LIBRARY_PATH'] = f"{lib_dir}:{current}" if current else lib_dir
                        return
                    except OSError:
                        continue

    logger.info("CUDA library not found in standard paths. Will use CPU.")


logger = logging.getLogger(__name__)
_setup_cuda_library_path()

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# ---------------------------------------------------------------------------
# Kalman Filter for bbox smoothing
# ---------------------------------------------------------------------------
class BboxKalmanFilter:
    """
    Simple Kalman filter for smoothing bounding box coordinates.
    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    """

    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.1):
        self.initialized = False
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.state = np.zeros(8, dtype=np.float64)
        # Covariance matrix
        self.P = np.eye(8, dtype=np.float64) * 10.0
        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0  # x1 += vx1
        self.F[1, 5] = 1.0  # y1 += vy1
        self.F[2, 6] = 1.0  # x2 += vx2
        self.F[3, 7] = 1.0  # y2 += vy2
        # Measurement matrix (observe position only)
        self.H = np.eye(4, 8, dtype=np.float64)
        # Process noise
        self.Q = np.eye(8, dtype=np.float64) * self.process_noise
        # Measurement noise
        self.R = np.eye(4, dtype=np.float64) * self.measurement_noise
        # Inactive frames counter
        self.inactive_frames = 0

    def predict(self) -> np.ndarray:
        """Predict next state, return smoothed bbox [x1, y1, x2, y2]."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.inactive_frames += 1
        return self.state[:4].copy()

    def update(self, measurement: np.ndarray):
        """Update with new measurement [x1, y1, x2, y2]."""
        z = np.asarray(measurement, dtype=np.float64)
        y = z - self.H @ self.state  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P
        self.inactive_frames = 0

    def get_bbox(self) -> np.ndarray:
        """Get current smoothed bbox."""
        return self.state[:4].copy()


@dataclass
class DetectionResult:
    """Single detection result with tracking ID."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2] normalized to 0-1
    frame_center_x: float
    frame_center_y: float
    tracker_id: Optional[int]  # Persistent ID from ByteTrack, None if tracking failed


@dataclass
class DetectionFrame:
    """Complete detection frame result."""
    detections: List[DetectionResult]
    object_counts: Dict[str, int]        # Per-frame counts
    session_counts: Dict[str, int]       # Unique object counts
    total_objects: int
    active_tracks_count: int
    total_unique_seen: int
    fps: float
    inference_ms: float
    capture_ms: float
    frame_width: int
    frame_height: int
    timestamp: float


class ObjectDetector:
    """
    YOLOv8 object detector with ByteTrack tracking.
    Each unique object gets a persistent tracker_id.
    """

    COLORS = [
        (56, 184, 235), (86, 207, 99), (247, 150, 70), (200, 80, 192),
        (130, 130, 230), (255, 130, 130), (130, 255, 130), (255, 200, 130),
        (200, 200, 200), (255, 255, 130),
    ]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        tracker: str = "bytetrack",
        models_dir: str = "",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_name = tracker
        self._models_dir = models_dir

        # Resolve tracker config: try models_dir/{tracker}_tuned.yaml first, then default
        self.tracker_config_path = f"{tracker}.yaml"  # fallback to ultralytics default
        if models_dir:
            tuned_path = os.path.join(models_dir, f"{tracker}_tuned.yaml")
            if os.path.isfile(tuned_path):
                self.tracker_config_path = tuned_path
            else:
                custom_path = os.path.join(models_dir, f"{tracker}.yaml")
                if os.path.isfile(custom_path):
                    self.tracker_config_path = custom_path
        logger.info(f"Tracker: {self.tracker_name} (config: {self.tracker_config_path})")

        import torch
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)

        if hasattr(self.model, 'names'):
            self.classes = self.model.names
        else:
            self.classes = {i: name for i, name in enumerate(COCO_CLASSES)}
        logger.info(f"Loaded {len(self.classes)} classes")

        # Performance tracking
        self.fps = 0.0
        self._fps_history: List[float] = []
        self._fps_window = 30
        self.total_frames_processed = 0
        self._start_time = time.time()

        # ── Tracking state ──────────────────────────────────────────────
        # Maps tracker_id → class_name (permanent for the session)
        self.known_tracks: Dict[int, str] = {}
        # Session counts: only incremented when a tracker_id is FIRST seen
        self.session_counts: Dict[str, int] = defaultdict(int)
        # Cumulative counts across restarts
        self.cumulative_counts: Dict[str, int] = defaultdict(int)

        # ── Stability metrics ──────────────────────────────────────────
        self.total_tracks_created: int = 0
        self.total_id_switches: int = 0
        # Maps tracker_id → set of class_names it has been assigned
        self._track_class_history: Dict[int, set] = defaultdict(set)

        # ── Bbox smoothing (Kalman filter per tracker_id) ─────────────
        self._kalman_filters: Dict[int, BboxKalmanFilter] = {}
        self._kalman_max_inactive = 60  # remove filter after N frames unseen

        # ── Confidence smoothing ───────────────────────────────────────
        self._conf_history: Dict[int, collections.deque] = defaultdict(
            lambda: collections.deque(maxlen=5)
        )

    def update_thresholds(self, conf_threshold: float, iou_threshold: float):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info(f"Thresholds: conf={conf_threshold}, iou={iou_threshold}")

    def set_tracker(self, tracker_name: str):
        """Switch tracker type at runtime (reinitializes Kalman filters)."""
        if tracker_name not in ("bytetrack", "botsort"):
            logger.warning(f"Unknown tracker: {tracker_name}")
            return
        self.tracker_name = tracker_name
        # Resolve config for the new tracker
        if hasattr(self, '_models_dir') and self._models_dir:
            tuned = os.path.join(self._models_dir, f"{tracker_name}_tuned.yaml")
            if os.path.isfile(tuned):
                self.tracker_config_path = tuned
            else:
                self.tracker_config_path = f"{tracker_name}.yaml"
        else:
            self.tracker_config_path = f"{tracker_name}.yaml"
        # Reset Kalman filters since tracking state changes
        self._kalman_filters.clear()
        logger.info(f"Tracker switched to {tracker_name} (config: {self.tracker_config_path})")

    def _get_smoothed_bbox(
        self, tracker_id: int, raw_bbox: np.ndarray, frame_w: int, frame_h: int
    ) -> List[float]:
        """Apply Kalman smoothing to bbox, return normalized [x1, y1, x2, y2]."""
        # Convert normalized to pixel space for Kalman (more stable numerically)
        px_bbox = np.array([
            raw_bbox[0] * frame_w,
            raw_bbox[1] * frame_h,
            raw_bbox[2] * frame_w,
            raw_bbox[3] * frame_h,
        ], dtype=np.float64)

        if tracker_id not in self._kalman_filters:
            kf = BboxKalmanFilter(process_noise=0.03, measurement_noise=0.08)
            kf.update(px_bbox)
            self._kalman_filters[tracker_id] = kf
            return list(raw_bbox)

        kf = self._kalman_filters[tracker_id]
        kf.predict()
        kf.update(px_bbox)

        smoothed_px = kf.get_bbox()
        # Clamp to frame bounds
        smoothed_px[0] = max(0, min(smoothed_px[0], frame_w))
        smoothed_px[1] = max(0, min(smoothed_px[1], frame_h))
        smoothed_px[2] = max(0, min(smoothed_px[2], frame_w))
        smoothed_px[3] = max(0, min(smoothed_px[3], frame_h))

        return [
            smoothed_px[0] / frame_w,
            smoothed_px[1] / frame_h,
            smoothed_px[2] / frame_w,
            smoothed_px[3] / frame_h,
        ]

    def _smooth_confidence(self, tracker_id: int, raw_conf: float) -> float:
        """Exponential moving average for confidence values."""
        history = self._conf_history[tracker_id]
        history.append(raw_conf)
        if len(history) == 1:
            return raw_conf
        # EMA with alpha=0.6 (favor recent, but smooth)
        alpha = 0.6
        ema = history[0]
        for c in list(history)[1:]:
            ema = alpha * c + (1 - alpha) * ema
        return ema

    def _cleanup_stale_filters(self, active_ids: set):
        """Remove Kalman filters for tracks that haven't been seen recently."""
        stale = [
            tid for tid, kf in self._kalman_filters.items()
            if tid not in active_ids and kf.inactive_frames > self._kalman_max_inactive
        ]
        for tid in stale:
            del self._kalman_filters[tid]
            self._conf_history.pop(tid, None)

    def detect(self, frame: np.ndarray) -> DetectionFrame:
        """
        Run tracking on a single frame.
        Each unique object gets a persistent tracker_id via ByteTrack.
        """
        frame_start = time.time()
        h, w = frame.shape[:2]

        # Run tracking (NOT detection) — persist=True keeps IDs across frames
        infer_start = time.time()
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            tracker=self.tracker_config_path,
            persist=True,
        )
        infer_time = (time.time() - infer_start) * 1000

        detections: List[DetectionResult] = []
        frame_counts: Dict[str, int] = defaultdict(int)
        current_frame_ids: set = set()

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        raw_conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.classes.get(class_id, f"class_{class_id}")

                        # Get tracker ID
                        tracker_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            tracker_id = int(box.id[0])

                        if tracker_id is not None:
                            current_frame_ids.add(tracker_id)

                            # ── ID switch detection ──────────────────
                            is_new_track = tracker_id not in self.known_tracks
                            if not is_new_track:
                                prev_class = self.known_tracks[tracker_id]
                                if prev_class != class_name:
                                    # ID switch detected!
                                    self.total_id_switches += 1
                                    logger.info(
                                        f"ID switch: track #{tracker_id} "
                                        f"'{prev_class}' -> '{class_name}' "
                                        f"(total switches: {self.total_id_switches})"
                                    )
                            else:
                                # New track
                                self.total_tracks_created += 1
                                logger.debug(
                                    f"New track #{tracker_id}: {class_name} "
                                    f"(total tracks: {self.total_tracks_created})"
                                )

                            # Update known tracks (always use latest class)
                            self.known_tracks[tracker_id] = class_name
                            self._track_class_history[tracker_id].add(class_name)

                            # ── Bbox smoothing ──────────────────────
                            raw_bbox = np.array([
                                x1 / w, y1 / h, x2 / w, y2 / h
                            ])
                            smoothed_bbox = self._get_smoothed_bbox(
                                tracker_id, raw_bbox, w, h
                            )

                            # ── Confidence smoothing ────────────────
                            smoothed_conf = self._smooth_confidence(tracker_id, raw_conf)

                            detection = DetectionResult(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=smoothed_conf,
                                bbox=smoothed_bbox,
                                frame_center_x=(smoothed_bbox[0] + smoothed_bbox[2]) / 2,
                                frame_center_y=(smoothed_bbox[1] + smoothed_bbox[3]) / 2,
                                tracker_id=tracker_id,
                            )
                            detections.append(detection)
                            frame_counts[class_name] += 1

                            # Count ONLY when this tracker_id is FIRST seen
                            if is_new_track:
                                self.session_counts[class_name] += 1
                                self.cumulative_counts[class_name] += 1

                        else:
                            # No tracker ID — fallback detection
                            detection = DetectionResult(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=raw_conf,
                                bbox=[x1 / w, y1 / h, x2 / w, y2 / h],
                                frame_center_x=(x1 + x2) / 2 / w,
                                frame_center_y=(y1 + y2) / 2 / h,
                                tracker_id=None,
                            )
                            detections.append(detection)
                            frame_counts[class_name] += 1
                            self.session_counts[class_name] += 1
                            self.cumulative_counts[class_name] += 1

                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue

        # ── Cleanup stale Kalman filters ──────────────────────────────
        self._cleanup_stale_filters(current_frame_ids)

        # Update FPS
        frame_time = (time.time() - frame_start) * 1000
        self._fps_history.append(1.0 / (frame_time / 1000) if frame_time > 0 else 0)
        if len(self._fps_history) > self._fps_window:
            self._fps_history.pop(0)
        current_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0
        self.fps = current_fps

        self.total_frames_processed += 1

        return DetectionFrame(
            detections=detections,
            object_counts=dict(frame_counts),
            session_counts=dict(self.session_counts),
            total_objects=len(detections),
            active_tracks_count=len(current_frame_ids),
            total_unique_seen=len(self.known_tracks),
            fps=current_fps,
            inference_ms=round(infer_time, 1),
            capture_ms=0.0,
            frame_width=w,
            frame_height=h,
            timestamp=time.time(),
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        show_labels: bool = True,
        show_conf: bool = True,
        show_tracker_id: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes with tracker IDs on frame."""
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = display.shape[:2]

        for det in detections:
            x1 = int(det.bbox[0] * w)
            y1 = int(det.bbox[1] * h)
            x2 = int(det.bbox[2] * w)
            y2 = int(det.bbox[3] * h)

            color = self.COLORS[det.class_id % len(self.COLORS)]

            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Build label: "person 0.92 #3"
            label_parts = []
            if show_labels:
                label_parts.append(det.class_name)
            if show_conf:
                label_parts.append(f"{det.confidence:.2f}")
            if show_tracker_id and det.tracker_id is not None:
                label_parts.append(f"#{det.tracker_id}")
            label = " ".join(label_parts)

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                display,
                (x1, y1 - label_h - 6),
                (x1 + label_w + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                display,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return display

    def reset_session_counts(self):
        """Reset session counts and tracking state."""
        self.session_counts.clear()
        self.known_tracks.clear()
        self._kalman_filters.clear()
        self._conf_history.clear()
        self._track_class_history.clear()
        self.total_tracks_created = 0
        self.total_id_switches = 0
        logger.info("Session counts, tracks, and stability metrics reset")

    def get_tracking_stability(self) -> float:
        """
        Compute tracking stability as 0.0-1.0.
        Uses minimum denominator of 3 to prevent extreme scores from small samples.
        stability = 1.0 - (id_switches / max(3, total_tracks))
        1.0 = perfect (no ID switches), 0.0 = heavy switching
        """
        total = max(3, self.total_tracks_created)
        return max(0.0, min(1.0, 1.0 - (self.total_id_switches / total)))

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            # Core metrics
            "fps": round(self.fps, 1),
            "frames_processed": self.total_frames_processed,
            "uptime_seconds": round(elapsed, 1),
            # Detection counts
            "session_counts": dict(self.session_counts),
            "cumulative_counts": dict(self.cumulative_counts),
            "total_unique_seen": len(self.known_tracks),
            "active_tracks": 0,
            "total_objects_detected": sum(self.session_counts.values()),
            "unique_classes_detected": len(self.session_counts),
            # Thresholds
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            # Tracker info (NEW — was missing)
            "tracker": self.tracker_name,
            "tracker_config": self.tracker_config_path,
            "total_tracks_created": self.total_tracks_created,
            "total_id_switches": self.total_id_switches,
            "tracking_stability": round(self.get_tracking_stability(), 4),
        }
