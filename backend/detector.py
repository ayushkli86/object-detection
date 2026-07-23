"""
YOLO Object Detector with Multi-Tracker Support
Uses Ultralytics YOLOv8 pretrained on COCO dataset (traffic classes).
Each unique object gets a persistent tracker_id across frames.

Improvements over v1:
  - YOLOv8l default model (52.9 mAP vs 37.3 nano)
  - Configurable class filtering (skip irrelevant COCO classes)
  - Pre-resize to imgsz before inference for consistent latency
  - Runtime model hot-swap (n/s/m/l/x without restart)
  - Adaptive Kalman filter per object type
  - Detection history ring-buffer for export
  - Zone/ROI filtering support
"""

import os
import csv
import io
import numpy as np
import ctypes
import time
import logging
import glob
import collections
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


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
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',       # 0-8
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',         # 9-15
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',                        # 16-23
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',          # 24-31
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',             # 32-37
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',              # 38-45
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',              # 46-53
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',            # 54-61
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',              # 62-69
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'hair drier',          # 70-77
    'toothbrush'                                                                                     # 78
]

 
CLASS_SUBSETS_COCO = {
    "all": list(range(79)),          # 0-78: all 79 COCO classes
    "traffic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # person, all vehicles, traffic infra
    "vehicles": [1, 2, 3, 4, 5, 6, 7, 8],   # bicycle, car, motorcycle, airplane, bus, train, truck, boat
    "people_animals": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # person, bird-giraffe
    "objects": list(range(24, 79)),   # backpack(24) through toothbrush(78)
}


def build_class_subsets(class_names: Dict[int, str]) -> Dict[str, List[int]]:
    """Build class subsets dynamically based on the model's class names.
    
    This ensures subsets work correctly regardless of which model is loaded
 
    """
    # Keywords that identify classes in each category
    traffic_kw = {
        'car', 'motorcycle', 'motorbike', 'bicycle', 'bike', 'bus', 'truck', 'train',
        'boat', 'airplane', 'vehicle', 'traffic light', 'stop sign', 'fire hydrant',
        'parking meter', 'tractor', 'auto', 'rickshaw', 'tempo',
    }
    vehicles_kw = {
        'car', 'motorcycle', 'motorbike', 'bicycle', 'bike', 'bus', 'truck',
        'train', 'boat', 'airplane', 'vehicle', 'tractor', 'auto', 'rickshaw', 'tempo',
    }
    people_kw = {
        'person', 'man', 'woman', 'child', 'boy', 'girl', 'pedestrian',
    }
    animals_kw = {
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'chicken', 'rabbit', 'animal',
    }
    # Furniture/electronics/daily objects — anything not traffic/people/animals
    object_kw = {
        'chair', 'table', 'desk', 'couch', 'sofa', 'bed', 'shelf', 'cabinet',
        'book', 'pen', 'pencil', 'notebook', 'paper', 'scissors', 'ruler',
        'laptop', 'computer', 'keyboard', 'mouse', 'monitor', 'cell phone',
        'telephone', 'tv', 'television', 'remote', 'printer',
        'bottle', 'cup', 'glass', 'bowl', 'plate', 'fork', 'knife', 'spoon',
        'refrigerator', 'microwave', 'oven', 'toaster', 'sink',
        'backpack', 'handbag', 'suitcase', 'umbrella', 'wallet',
        'hat', 'shoe', 'shirt', 'jacket', 'pants',
        'clock', 'vase', 'hammer', 'screwdriver',
    }

    subsets: Dict[str, List[int]] = {"all": list(class_names.keys())}
    seen: Dict[str, set] = {k: set() for k in ("traffic", "vehicles", "people_animals", "animals", "objects")}
    for idx, name in class_names.items():
        name_lower = name.lower().strip()
        if name_lower in traffic_kw:
            if idx not in seen["traffic"]:
                subsets.setdefault("traffic", []).append(idx)
                seen["traffic"].add(idx)
            if idx not in seen["vehicles"]:
                subsets.setdefault("vehicles", []).append(idx)
                seen["vehicles"].add(idx)
        if name_lower in vehicles_kw and idx not in seen["vehicles"]:
            subsets.setdefault("vehicles", []).append(idx)
            seen["vehicles"].add(idx)
        if name_lower in people_kw and idx not in seen["people_animals"]:
            subsets.setdefault("people_animals", []).append(idx)
            seen["people_animals"].add(idx)
        if name_lower in animals_kw:
            if idx not in seen["people_animals"]:
                subsets.setdefault("people_animals", []).append(idx)
                seen["people_animals"].add(idx)
            if idx not in seen["animals"]:
                subsets.setdefault("animals", []).append(idx)
                seen["animals"].add(idx)
        if name_lower in object_kw or (
            # catch-all: any object not in traffic/people/animals
            name_lower not in traffic_kw
            and name_lower not in people_kw
            and name_lower not in animals_kw
            and any(kw in name_lower for kw in ['object', 'item', 'thing', 'furniture', 'electronic', 'food', 'fruit', 'vegetable', 'clothing', 'sport'])
        ):
            if idx not in seen["objects"]:
                subsets.setdefault("objects", []).append(idx)
                seen["objects"].add(idx)

    # Always ensure these subsets exist even if empty
    for key in ("traffic", "vehicles", "people_animals", "animals", "objects"):
        subsets.setdefault(key, [])

    return subsets

# Available model sizes (YOLOv8)
AVAILABLE_MODELS = {
    "besst": {"file": "besst.pt", "params": "43.7M", "map": 52.9, "speed_ms": 2.39},
    "yolov8l-oiv7": {"file": "yolov8l-oiv7.pt", "params": "44.1M", "map": 34.9, "speed_ms": 2.43, "description": "601 classes (traffic + daily objects)", "finetuned": True},
}


# ---------------------------------------------------------------------------
# Kalman Filter for bbox smoothing
# ---------------------------------------------------------------------------
class BboxKalmanFilter:
    """
    Kalman filter for smoothing bounding box coordinates.
    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    """

    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.1):
        self.initialized = False
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = np.zeros(8, dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0
        self.H = np.eye(4, 8, dtype=np.float64)
        self.Q = np.eye(8, dtype=np.float64) * self.process_noise
        self.R = np.eye(4, dtype=np.float64) * self.measurement_noise
        self.inactive_frames = 0

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.inactive_frames += 1
        return self.state[:4].copy()

    def update(self, measurement: np.ndarray):
        z = np.asarray(measurement, dtype=np.float64)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix — use pseudo-inverse as fallback
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.state = self.state + K @ y
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P
        self.inactive_frames = 0

    def get_bbox(self) -> np.ndarray:
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
    tracker_id: Optional[int]
    tracking_duration_frames: int = 0
    first_seen_timestamp: float = 0.0
    zone_id: Optional[int] = None  # ROI zone this detection falls in


@dataclass
class DetectionFrame:
    """Complete detection frame result."""
    detections: List[DetectionResult]
    object_counts: Dict[str, int]
    session_counts: Dict[str, int]
    total_objects: int
    active_tracks_count: int
    total_unique_seen: int
    fps: float
    inference_ms: float
    frame_width: int
    frame_height: int
    model_name: str = "besst"


@dataclass
class DetectionZone:
    """A region-of-interest polygon for selective detection."""
    id: int
    name: str
    points: List[List[int]]  # [[x1,y1], [x2,y2], ...] in normalized 0-1 coords
    enabled: bool = True


@dataclass
class HistoryEntry:
    """A single detection history record."""
    timestamp: float
    frame_width: int
    frame_height: int
    detections_count: int
    object_counts: Dict[str, int]
    active_tracks: int
    fps: float
    inference_ms: float


class ObjectDetector:
    """
    YOLO object detector with multi-tracker support.
    Each unique object gets a persistent tracker_id.
    """

    # Motion-adaptive Kalman noise per category
    CATEGORY_MOTION_NOISE = {
        'car': 0.05, 'truck': 0.04, 'bus': 0.03, 'motorcycle': 0.07,
        'bicycle': 0.06, 'person': 0.08,
    }

    def __init__(
        self,
        model_path: str = "besst.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        tracker: str = "bytetrack",
        models_dir: str = "",
        imgsz: int = 640,
        class_subset: str = "all",
        max_history: int = 10000,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_name = tracker
        self._models_dir = models_dir
        self.imgsz = imgsz
        self._max_history = max_history

        # ── Class filtering ────────────────────────────────────────────
        self._class_subset_name = class_subset
        self._class_subsets = {}  # Built after model loads below
        self._filtered_classes = list(range(79))  # temporary, rebuilt after model load
        logger.info(f"Class filter: {class_subset} ({len(self._filtered_classes)} classes)")

        # Resolve tracker config
        self.tracker_config_path = f"{tracker}.yaml"
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

        # ── Load model ─────────────────────────────────────────────────
        self.model_name = "besst"
        self._model_path = model_path
        logger.info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

        if hasattr(self.model, 'names'):
            self.classes = self.model.names
        else:
            self.classes = {i: name for i, name in enumerate(COCO_CLASSES)}
        logger.info(f"Loaded {len(self.classes)} classes")

        # ── Build dynamic class subsets for this model ───────────────────
        self._class_subsets = build_class_subsets(self.classes)
        logger.info(f"Built {len(self._class_subsets)} class subsets: {list(self._class_subsets.keys())}")
        if class_subset in self._class_subsets:
            self._filtered_classes = self._class_subsets[class_subset]
        else:
            self._filtered_classes = list(self.classes.keys())
        logger.info(f"Class filter: {class_subset} ({len(self._filtered_classes)} classes)")

        # Performance tracking
        self.fps = 0.0
        self._fps_history: collections.deque = collections.deque(maxlen=30)
        self.total_frames_processed = 0
        self._start_time = time.time()

        # ── Tracking state ──────────────────────────────────────────────
        self.known_tracks: Dict[int, str] = {}
        self.session_counts: Dict[str, int] = defaultdict(int)
        self.cumulative_counts: Dict[str, int] = defaultdict(int)

        # ── Stability metrics ──────────────────────────────────────────
        self.total_tracks_created: int = 0
        self.total_id_switches: int = 0

        # ── Bbox smoothing (Kalman filter per tracker_id) ─────────────
        self._kalman_filters: Dict[int, BboxKalmanFilter] = {}
        self._kalman_max_inactive = 60

        # ── Confidence smoothing ───────────────────────────────────────
        self._conf_history: Dict[int, collections.deque] = defaultdict(
            lambda: collections.deque(maxlen=5)
        )

        # ── Track timestamps & frame counts ────────────────────────────
        self._track_first_seen: Dict[int, float] = {}
        self._track_frame_count: Dict[int, int] = defaultdict(int)

        # ── Re-identification: lost-track buffer ───────────────────────
        # Maps old tracker_id → metadata for recently disappeared tracks.
        # When a new tracker_id appears, we check this buffer for a spatial
        # match (IoU) to reassign the old ID instead of creating a duplicate.
        self._lost_tracks: Dict[int, Dict[str, Any]] = {}
        self._lost_track_max_age: float = 3.0  # seconds before purging

        # ── Untracked-detection frame-to-frame matching ────────────────
        # IoU-based matching prevents the same untracked person from being
        # counted fresh every frame.
        self._untracked_prev: List[Dict[str, Any]] = []

        # ── Detection history ring-buffer ──────────────────────────────
        self._history: collections.deque = collections.deque(maxlen=max_history)

        # ── Zones / ROI ────────────────────────────────────────────────
        self._zones: List[DetectionZone] = []
        self._next_zone_id = 1

    # ── Model hot-swap ─────────────────────────────────────────────────
    def switch_model(self, model_name: str) -> bool:
        """Hot-swap the YOLO model at runtime. Returns True on success."""
        if model_name not in AVAILABLE_MODELS:
            logger.warning(f"Unknown model: {model_name}")
            return False

        info = AVAILABLE_MODELS[model_name]
        model_file = info["file"]

        # Resolve path: check models_dir first, then CWD, then let ultralytics download
        resolved = model_file
        if self._models_dir:
            full_path = os.path.join(self._models_dir, model_file)
            if os.path.isfile(full_path):
                resolved = full_path

        try:
            logger.info(f"Switching model to {model_name} ({info['params']} params, mAP {info['map']})...")
            # Load into temporary variable first to avoid partial state
            new_model = YOLO(resolved)
            self.model = new_model
            self.model_name = model_name
            self._model_path = resolved

            if hasattr(self.model, 'names'):
                self.classes = self.model.names
            else:
                self.classes = {i: name for i, name in enumerate(COCO_CLASSES)}

            # Rebuild class subsets for new model
            self._class_subsets = build_class_subsets(self.classes)
            if self._class_subset_name in self._class_subsets:
                self._filtered_classes = self._class_subsets[self._class_subset_name]
            else:
                self._filtered_classes = list(self.classes.keys())

            # Reset tracking state since model changed
            self._kalman_filters.clear()
            self._conf_history.clear()
            logger.info(f"Model switched to {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    # ── Class filtering ────────────────────────────────────────────────
    def set_class_filter(self, subset: str):
        """Change the active class filter using dynamic model-aware subsets."""
        if subset in self._class_subsets:
            self._class_subset_name = subset
            self._filtered_classes = self._class_subsets[subset]
            logger.info(f"Class filter: {subset} ({len(self._filtered_classes)} classes)")
        elif subset == "all":
            self._class_subset_name = "all"
            self._filtered_classes = list(self.classes.keys())
            logger.info(f"Class filter: all ({len(self._filtered_classes)} classes)")
        else:
            logger.warning(f"Unknown class subset '{subset}' for current model")

    # ── Zone management ────────────────────────────────────────────────
    def add_zone(self, name: str, points: List[List[int]]) -> DetectionZone:
        zone = DetectionZone(id=self._next_zone_id, name=name, points=points)
        self._next_zone_id += 1
        self._zones.append(zone)
        logger.info(f"Zone added: {name} (id={zone.id})")
        return zone

    def remove_zone(self, zone_id: int) -> bool:
        before = len(self._zones)
        self._zones = [z for z in self._zones if z.id != zone_id]
        return len(self._zones) < before

    def clear_zones(self):
        self._zones.clear()

    @staticmethod
    def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes (normalized or pixel)."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _point_in_polygon(self, px: float, py: float, polygon: List[List[int]]) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _get_zone_for_point(self, cx: float, cy: float) -> Optional[int]:
        """Returns the zone_id if point falls in any active zone, or None."""
        active_zones = [z for z in self._zones if z.enabled]
        if not active_zones:
            return None
        for zone in active_zones:
            if self._point_in_polygon(cx, cy, zone.points):
                return zone.id
        return -1  # Not in any zone = excluded

    # ── Existing methods (updated) ─────────────────────────────────────
    def update_thresholds(self, conf_threshold: float, iou_threshold: float):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info(f"Thresholds: conf={conf_threshold}, iou={iou_threshold}")

    def set_tracker(self, tracker_name: str):
        if tracker_name not in ("bytetrack", "botsort"):
            logger.warning(f"Unknown tracker: {tracker_name}")
            return
        self.tracker_name = tracker_name
        if hasattr(self, '_models_dir') and self._models_dir:
            tuned = os.path.join(self._models_dir, f"{tracker_name}_tuned.yaml")
            if os.path.isfile(tuned):
                self.tracker_config_path = tuned
            else:
                self.tracker_config_path = f"{tracker_name}.yaml"
        else:
            self.tracker_config_path = f"{tracker_name}.yaml"
        self._kalman_filters.clear()
        logger.info(f"Tracker switched to {tracker_name}")

    def _get_smoothed_bbox(
        self, tracker_id: int, raw_bbox: np.ndarray, frame_w: int, frame_h: int,
        class_name: str = "",
    ) -> List[float]:
        """Apply Kalman smoothing to bbox with class-adaptive noise."""
        px_bbox = np.array([
            raw_bbox[0] * frame_w, raw_bbox[1] * frame_h,
            raw_bbox[2] * frame_w, raw_bbox[3] * frame_h,
        ], dtype=np.float64)

        if tracker_id not in self._kalman_filters:
            noise = self.CATEGORY_MOTION_NOISE.get(class_name, 0.03)
            kf = BboxKalmanFilter(process_noise=noise, measurement_noise=0.08)
            kf.update(px_bbox)
            self._kalman_filters[tracker_id] = kf
            return list(raw_bbox)

        kf = self._kalman_filters[tracker_id]
        kf.predict()
        kf.update(px_bbox)
        smoothed_px = kf.get_bbox()
        smoothed_px[0] = max(0, min(smoothed_px[0], frame_w))
        smoothed_px[1] = max(0, min(smoothed_px[1], frame_h))
        smoothed_px[2] = max(0, min(smoothed_px[2], frame_w))
        smoothed_px[3] = max(0, min(smoothed_px[3], frame_h))
        return [smoothed_px[0] / frame_w, smoothed_px[1] / frame_h,
                smoothed_px[2] / frame_w, smoothed_px[3] / frame_h]

    def _smooth_confidence(self, tracker_id: int, raw_conf: float) -> float:
        history = self._conf_history[tracker_id]
        history.append(raw_conf)
        if len(history) == 1:
            return raw_conf
        alpha = 0.6
        ema = history[0]
        for c in list(history)[1:]:
            ema = alpha * c + (1 - alpha) * ema
        return ema

    def _cleanup_stale_filters(self, active_ids: set):
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
        Supports class filtering, zone filtering, and configurable imgsz.
        """
        frame_start = time.time()
        h, w = frame.shape[:2]

        # Run tracking with class filtering
        infer_start = time.time()
        kwargs = dict(
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            tracker=self.tracker_config_path,
            persist=True,
            imgsz=self.imgsz,
        )
        # Apply class filtering if not "all"
        if self._class_subset_name != "all" and len(self._filtered_classes) < len(self.classes):
            kwargs["classes"] = self._filtered_classes

        results = self.model.track(frame, **kwargs)
        infer_time = (time.time() - infer_start) * 1000

        detections: List[DetectionResult] = []
        frame_counts: Dict[str, int] = defaultdict(int)
        current_frame_ids: set = set()
        zones_active = len(self._zones) > 0 and any(z.enabled for z in self._zones)
        now = time.time()

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # ── Phase 1: Collect all raw detections from YOLO ──────
                raw_detections: List[Dict[str, Any]] = []
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        raw_conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.classes.get(class_id, f"class_{class_id}")

                        # Zone filtering
                        center_x = (x1 + x2) / 2 / w
                        center_y = (y1 + y2) / 2 / h
                        zone_id = None
                        if zones_active:
                            zid = self._get_zone_for_point(center_x, center_y)
                            if zid == -1:
                                continue
                            zone_id = zid

                        tracker_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            tracker_id = int(box.id[0])

                        raw_detections.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'raw_conf': raw_conf, 'class_id': class_id,
                            'class_name': class_name, 'center_x': center_x,
                            'center_y': center_y, 'zone_id': zone_id,
                            'tracker_id': tracker_id,
                            'norm_bbox': [x1 / w, y1 / h, x2 / w, y2 / h],
                        })
                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue

                # ── Phase 2: Remember previous tracked IDs ────────────────
                prev_tracked_ids = set(self.known_tracks.keys())

                # ── Phase 3: Process tracked detections with re-ID ──────
                used_lost_ids: set = set()  # lost-track IDs already re-assigned this frame

                for rd in raw_detections:
                    tracker_id = rd['tracker_id']

                    if tracker_id is not None:
                        current_frame_ids.add(tracker_id)
                        is_new_track = tracker_id not in self.known_tracks

                        # ── RE-IDENTIFICATION: check lost-track buffer ──
                        if is_new_track:
                            # Try to find a matching lost track of the same class
                            best_lost_id = None
                            best_iou = 0.0
                            for lost_id, lost_info in self._lost_tracks.items():
                                if lost_id in used_lost_ids:
                                    continue
                                if lost_info['class_name'] != rd['class_name']:
                                    continue
                                iou = self._compute_iou(rd['norm_bbox'], lost_info['bbox'])
                                if iou > best_iou and iou >= 0.25:  # relaxed threshold for re-ID
                                    best_iou = iou
                                    best_lost_id = lost_id

                            if best_lost_id is not None:
                                # RE-IDENTIFIED: reuse old tracker_id
                                old_id = best_lost_id
                                lost_info = self._lost_tracks.pop(old_id)
                                used_lost_ids.add(old_id)

                                # Remove old Kalman filter so it re-initializes
                                self._kalman_filters.pop(old_id, None)

                                # Merge: keep the OLD tracker_id, restore metadata
                                # The YOLO tracker gave us a new ID, but we override it
                                # by removing the new ID from current_frame_ids and using old
                                current_frame_ids.discard(tracker_id)
                                current_frame_ids.add(old_id)
                                rd['tracker_id'] = old_id
                                tracker_id = old_id

                                # Restore track metadata from lost buffer
                                self.known_tracks[old_id] = rd['class_name']
                                self._track_frame_count[old_id] = lost_info.get('track_frame_count', 0)
                                self._track_first_seen[old_id] = lost_info.get('first_seen', now)

                                logger.debug(
                                    f"Re-identified track #{old_id} "
                                    f"'{rd['class_name']}' (IoU={best_iou:.2f})"
                                )
                                is_new_track = False
                            # else: genuinely new track, no match found
                        # END re-identification

                        if not is_new_track:
                            prev_class = self.known_tracks[tracker_id]
                            if prev_class != rd['class_name']:
                                self.total_id_switches += 1
                                logger.info(
                                    f"ID switch: track #{tracker_id} "
                                    f"'{prev_class}' -> '{rd['class_name']}' "
                                    f"(total: {self.total_id_switches})"
                                )
                        else:
                            self.total_tracks_created += 1

                        self.known_tracks[tracker_id] = rd['class_name']
                        self._track_frame_count[tracker_id] += 1

                        # Bbox smoothing
                        raw_bbox = np.array(rd['norm_bbox'])
                        smoothed_bbox = self._get_smoothed_bbox(
                            tracker_id, raw_bbox, w, h, rd['class_name']
                        )
                        smoothed_conf = self._smooth_confidence(tracker_id, rd['raw_conf'])

                        detection = DetectionResult(
                            class_id=rd['class_id'],
                            class_name=rd['class_name'],
                            confidence=smoothed_conf,
                            bbox=smoothed_bbox,
                            frame_center_x=rd['center_x'],
                            frame_center_y=rd['center_y'],
                            tracker_id=tracker_id,
                            tracking_duration_frames=self._track_frame_count[tracker_id],
                            first_seen_timestamp=self._track_first_seen.get(tracker_id, now),
                            zone_id=rd['zone_id'],
                        )
                        detections.append(detection)
                        frame_counts[rd['class_name']] += 1

                        if is_new_track:
                            self.session_counts[rd['class_name']] += 1
                            self.cumulative_counts[rd['class_name']] += 1

                # ── Phase 4: Moved-to-lost for tracks that disappeared ──
                disappeared_ids = prev_tracked_ids - current_frame_ids
                for old_id in disappeared_ids:
                    if old_id in self.known_tracks:
                        self._lost_tracks[old_id] = {
                            'bbox': self._kalman_filters[old_id].get_bbox().tolist()
                                    if old_id in self._kalman_filters else [0, 0, 0, 0],
                            'class_name': self.known_tracks[old_id],
                            'lost_at': now,
                            'track_frame_count': self._track_frame_count.get(old_id, 0),
                            'first_seen': self._track_first_seen.get(old_id, now),
                        }

                # ── Phase 5: Untracked detections with spatial matching ──
                # Match untracked detections against previous frame's untracked
                # to avoid counting the same person every frame.
                current_untracked: List[Dict[str, Any]] = []
                for rd in raw_detections:
                    if rd['tracker_id'] is not None:
                        continue  # already processed above

                    matched_prev = False
                    best_idx = -1
                    best_iou = 0.0
                    for pi, prev in enumerate(self._untracked_prev):
                        if prev.get('consumed', False):
                            continue
                        if prev['class_name'] != rd['class_name']:
                            continue
                        iou = self._compute_iou(rd['norm_bbox'], prev['bbox'])
                        if iou > best_iou and iou >= 0.3:
                            best_iou = iou
                            best_idx = pi

                    if best_idx >= 0:
                        # Same untracked object from previous frame — reuse
                        self._untracked_prev[best_idx]['consumed'] = True
                        matched_prev = True

                    detection = DetectionResult(
                        class_id=rd['class_id'],
                        class_name=rd['class_name'],
                        confidence=rd['raw_conf'],
                        bbox=rd['norm_bbox'],
                        frame_center_x=rd['center_x'],
                        frame_center_y=rd['center_y'],
                        tracker_id=None,
                        zone_id=rd['zone_id'],
                    )
                    detections.append(detection)
                    frame_counts[rd['class_name']] += 1

                    current_untracked.append({
                        'bbox': rd['norm_bbox'],
                        'class_name': rd['class_name'],
                        'consumed': False,
                    })

                    if not matched_prev:
                        # Genuinely new untracked object — count it once
                        self.session_counts[rd['class_name']] += 1
                        self.cumulative_counts[rd['class_name']] += 1

                self._untracked_prev = current_untracked

        # ── Phase 6: Cleanup ─────────────────────────────────────────
        # Purge stale lost tracks (>3s)
        stale_lost = [
            lid for lid, info in self._lost_tracks.items()
            if now - info['lost_at'] > self._lost_track_max_age
        ]
        for lid in stale_lost:
            del self._lost_tracks[lid]

        # Cleanup stale Kalman filters
        self._cleanup_stale_filters(current_frame_ids)

        # Update FPS
        frame_time = (time.time() - frame_start) * 1000
        self._fps_history.append(1.0 / (frame_time / 1000) if frame_time > 0 else 0)
        current_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0
        self.fps = current_fps
        self.total_frames_processed += 1

        # ── Record history entry ─────────────────────────────────────
        self._history.append(HistoryEntry(
            timestamp=time.time(),
            frame_width=w,
            frame_height=h,
            detections_count=len(detections),
            object_counts=dict(frame_counts),
            active_tracks=len(current_frame_ids),
            fps=round(current_fps, 1),
            inference_ms=round(infer_time, 1),
        ))

        return DetectionFrame(
            detections=detections,
            object_counts=dict(frame_counts),
            session_counts=dict(self.session_counts),
            total_objects=len(detections),
            active_tracks_count=len(current_frame_ids),
            total_unique_seen=len(self.known_tracks),
            fps=current_fps,
            inference_ms=round(infer_time, 1),
            frame_width=w,
            frame_height=h,
            model_name=self.model_name,
        )

    # ── History export ─────────────────────────────────────────────────
    def export_history_csv(self) -> str:
        """Export detection history as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "timestamp", "frame_width", "frame_height", "detections_count",
            "active_tracks", "fps", "inference_ms", "top_classes"
        ])
        for entry in self._history:
            top = ";".join(f"{k}:{v}" for k, v in
                          sorted(entry.object_counts.items(), key=lambda x: -x[1])[:5])
            writer.writerow([
                entry.timestamp, entry.frame_width, entry.frame_height,
                entry.detections_count, entry.active_tracks,
                entry.fps, entry.inference_ms, top,
            ])
        return output.getvalue()

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get recent detection history as list of dicts."""
        entries = list(self._history)[-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "detections_count": e.detections_count,
                "object_counts": e.object_counts,
                "active_tracks": e.active_tracks,
                "fps": e.fps,
                "inference_ms": e.inference_ms,
            }
            for e in entries
        ]

    def reset_session_counts(self):
        # Assign new empty containers instead of .clear() to avoid
        # RuntimeError: dictionary changed size during iteration
        self.session_counts = defaultdict(int)
        self.cumulative_counts = defaultdict(int)
        self.known_tracks = {}
        self._kalman_filters = {}
        self._conf_history = defaultdict(lambda: collections.deque(maxlen=5))
        self._track_frame_count = defaultdict(int)
        self._track_first_seen = {}
        self._lost_tracks = {}
        self._untracked_prev = []
        self.total_tracks_created = 0
        self.total_id_switches = 0
        logger.info("Session counts, tracks, and stability metrics reset")

    def get_tracking_stability(self) -> float:
        total = max(3, self.total_tracks_created)
        return max(0.0, min(1.0, 1.0 - (self.total_id_switches / total)))

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            "fps": round(self.fps, 1),
            "frames_processed": self.total_frames_processed,
            "uptime_seconds": round(elapsed, 1),
            "session_counts": dict(self.session_counts),
            "cumulative_counts": dict(self.cumulative_counts),
            "total_unique_seen": len(self.known_tracks),
            "active_tracks": 0,
            "total_objects_detected": sum(self.session_counts.values()),
            "unique_classes_detected": len(self.session_counts),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "tracker": self.tracker_name,
            "tracker_config": self.tracker_config_path,
            "total_tracks_created": self.total_tracks_created,
            "total_id_switches": self.total_id_switches,
            "tracking_stability": round(self.get_tracking_stability(), 4),
            # ── New fields ─────────────────────────────────────────────
            "model": self.model_name,
            "model_info": AVAILABLE_MODELS.get(self.model_name, {}),
            "imgsz": self.imgsz,
            "class_subset": self._class_subset_name,
            "active_classes": len(self._filtered_classes),
            "zones_count": len(self._zones),
            "history_count": len(self._history),
        }
