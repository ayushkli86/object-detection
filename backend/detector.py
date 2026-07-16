"""
YOLOv8 Object Detector with ByteTrack Tracking
Uses Ultralytics YOLOv8 pretrained on COCO dataset (80 classes)
Each unique object gets a persistent tracker_id across frames.
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
    object_counts: Dict[str, int]        # Per-frame counts (how many of each class in THIS frame)
    session_counts: Dict[str, int]       # Unique object counts (each tracker_id counted once)
    total_objects: int                    # Total detections in this frame
    active_tracks_count: int             # Unique objects currently visible
    total_unique_seen: int               # Total unique objects seen this session
    fps: float
    inference_ms: float
    capture_ms: float
    frame_width: int
    frame_height: int
    timestamp: float


class ObjectDetector:
    """
    YOLOv8 object detector with configurable tracker (BoT-SORT or ByteTrack).
    Each unique object gets a persistent tracker_id.
    """

    COLORS = [
        (56, 184, 235), (86, 207, 99), (247, 150, 70), (200, 80, 192),
        (130, 130, 230), (255, 130, 130), (130, 255, 130), (255, 200, 130),
        (200, 200, 200), (255, 255, 130),
    ]

    # Available trackers — path relative to models/ dir
    TRACKERS = {
        "botsort":     "botsort_tuned.yaml",     # Best: appearance + motion (ReID)
        "bytetrack":   "bytetrack_tuned.yaml",   # Lightweight: IoU only
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        tracker: str = "botsort",
        models_dir: str = "models",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.models_dir = models_dir

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

        # ── Tracker selection ───────────────────────────────────────────
        self.tracker_name = tracker
        self.tracker_config = self._resolve_tracker(tracker)
        logger.info(f"Tracker: {self.tracker_name} ({self.tracker_config})")

        # Performance tracking
        self.fps = 0.0
        self._fps_history: List[float] = []
        self._fps_window = 30
        self.total_frames_processed = 0
        self._start_time = time.time()

        # ── Tracking state ──────────────────────────────────────────────
        # Maps tracker_id → class_name (permanent for the session)
        self.known_tracks: Dict[int, str] = {}
        # Track lifetime: tracker_id → frames_seen
        self.track_lifetimes: Dict[int, int] = defaultdict(int)
        # Session counts: only incremented when a tracker_id is FIRST seen
        self.session_counts: Dict[str, int] = defaultdict(int)
        # Cumulative counts across restarts
        self.cumulative_counts: Dict[str, int] = defaultdict(int)
        # Tracking quality metrics
        self.total_id_switches = 0
        self.total_tracks_created = 0
        self._prev_frame_ids: set = set()

    def _resolve_tracker(self, tracker_name: str) -> str:
        """Resolve tracker name to config path."""
        if tracker_name in self.TRACKERS:
            config_path = os.path.join(self.models_dir, self.TRACKERS[tracker_name])
            if os.path.exists(config_path):
                return config_path
            logger.warning(f"Tracker config not found: {config_path}, falling back to default")
            return self.TRACKERS[tracker_name]  # Let ultralytics resolve defaults

        # Allow passing a direct path
        if os.path.exists(tracker_name):
            return tracker_name

        logger.warning(f"Unknown tracker '{tracker_name}', using bytetrack default")
        return "bytetrack.yaml"

    def set_tracker(self, tracker_name: str) -> str:
        """Switch tracker at runtime. Returns the resolved config path."""
        self.tracker_name = tracker_name
        self.tracker_config = self._resolve_tracker(tracker_name)
        # Reset tracking state when switching trackers
        self.known_tracks.clear()
        self.track_lifetimes.clear()
        self._prev_frame_ids.clear()
        self.total_id_switches = 0
        self.total_tracks_created = 0
        logger.info(f"Tracker switched to: {self.tracker_name} ({self.tracker_config})")
        return self.tracker_config

    def update_thresholds(self, conf_threshold: float, iou_threshold: float):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info(f"Thresholds: conf={conf_threshold}, iou={iou_threshold}")

    def detect(self, frame: np.ndarray) -> DetectionFrame:
        """
        Run tracking on a single frame.
        Uses selected tracker (BoT-SORT/ByteTrack) with persistent IDs.
        """
        frame_start = time.time()
        h, w = frame.shape[:2]

        # Run tracking with selected tracker config
        infer_start = time.time()
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            tracker=self.tracker_config,
            persist=True,
        )
        infer_time = (time.time() - infer_start) * 1000

        detections: List[DetectionResult] = []
        frame_counts: Dict[str, int] = defaultdict(int)

        # Track which IDs are in this frame for stability metrics
        current_frame_ids: set = set()

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.classes.get(class_id, f"class_{class_id}")

                        # Get tracker ID (None if tracking is off or failed)
                        tracker_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            tracker_id = int(box.id[0])

                        detection = DetectionResult(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=[x1 / w, y1 / h, x2 / w, y2 / h],
                            frame_center_x=(x1 + x2) / 2 / w,
                            frame_center_y=(y1 + y2) / 2 / h,
                            tracker_id=tracker_id,
                        )
                        detections.append(detection)
                        frame_counts[class_name] += 1

                        # ── Counting logic: only count NEW tracker_ids ───
                        if tracker_id is not None:
                            current_frame_ids.add(tracker_id)
                            if tracker_id not in self.known_tracks:
                                # First time seeing this object — count it
                                self.known_tracks[tracker_id] = class_name
                                self.track_lifetimes[tracker_id] = 1
                                self.session_counts[class_name] += 1
                                self.cumulative_counts[class_name] += 1
                                self.total_tracks_created += 1
                                logger.debug(
                                    f"New track #{tracker_id}: {class_name} "
                                    f"(total unique: {len(self.known_tracks)})"
                                )
                            else:
                                self.track_lifetimes[tracker_id] += 1
                        else:
                            # No tracker ID — count as frame detection (fallback)
                            self.session_counts[class_name] += 1
                            self.cumulative_counts[class_name] += 1

                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue

        # ── ID stability metric ─────────────────────────────────────────
        # Count tracks that were in previous frame but not in current frame
        # (not just temporarily occluded, but truly lost)
        if self._prev_frame_ids:
            lost_ids = self._prev_frame_ids - current_frame_ids
            self.total_id_switches += len(lost_ids)
        self._prev_frame_ids = current_frame_ids

        # Update FPS
        frame_time = (time.time() - frame_start) * 1000
        self._fps_history.append(1.0 / (frame_time / 1000) if frame_time > 0 else 0)
        if len(self._fps_history) > self._fps_window:
            self._fps_history.pop(0)
        current_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0
        self.fps = current_fps

        self.total_frames_processed += 1

        # Count currently active tracks (objects visible in this frame)
        active_ids = set()
        for det in detections:
            if det.tracker_id is not None:
                active_ids.add(det.tracker_id)

        return DetectionFrame(
            detections=detections,
            object_counts=dict(frame_counts),
            session_counts=dict(self.session_counts),
            total_objects=len(detections),
            active_tracks_count=len(active_ids),
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
        logger.info("Session counts and tracks reset")

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        # Tracking stability: tracks created vs ID switches
        stability = (
            round(1.0 - (self.total_id_switches / max(1, self.total_tracks_created)), 3)
            if self.total_tracks_created > 0 else 1.0
        )
        return {
            "fps": round(self.fps, 1),
            "frames_processed": self.total_frames_processed,
            "uptime_seconds": round(elapsed, 1),
            "session_counts": dict(self.session_counts),
            "cumulative_counts": dict(self.cumulative_counts),
            "total_unique_seen": len(self.known_tracks),
            "active_tracks": 0,  # Updated by caller after detect()
            "total_objects_detected": sum(self.session_counts.values()),
            "unique_classes_detected": len(self.session_counts),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "tracker": self.tracker_name,
            "tracker_config": self.tracker_config,
            "total_tracks_created": self.total_tracks_created,
            "total_id_switches": self.total_id_switches,
            "tracking_stability": stability,  # 1.0 = perfect, lower = more ID switches
        }
