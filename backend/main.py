"""
Object Detection - Backend Server (Camera Only)
FastAPI server for real-time YOLO camera object detection.
Single endpoint: POST /api/detect — accepts base64 JPEG, returns detections.
"""

import os
import sys
import logging
import time
import base64
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import ObjectDetector, AVAILABLE_MODELS, CLASS_SUBSETS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("camera-detector")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
TRACKER = os.environ.get("TRACKER", "botsort")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
CLASS_FILTER = os.environ.get("CLASS_FILTER", "all")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
RATE_LIMIT_DETECT_PER_SEC = int(os.environ.get("RATE_LIMIT_DETECT", "30"))


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_per_sec: int):
        self.max_per_sec = max_per_sec
        self._timestamps: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 1.0]
        if len(self._timestamps) >= self.max_per_sec:
            return False
        self._timestamps.append(now)
        return True

_rate_limiter = RateLimiter(RATE_LIMIT_DETECT_PER_SEC)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ConfigUpdate(BaseModel):
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    max_fps: Optional[int] = None
    tracker: Optional[str] = None
    model: Optional[str] = None
    imgsz: Optional[int] = None
    class_filter: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Starting Camera Object Detector backend...")

    # Resolve model path
    resolved_path = ""
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        resolved_path = MODEL_PATH
    elif MODELS_DIR:
        for candidate in ["yolov8l.pt", "yolov8n.pt"]:
            full = os.path.join(MODELS_DIR, candidate)
            if os.path.isfile(full):
                resolved_path = full
                break

    if not resolved_path:
        resolved_path = "yolov8l.pt"
        logger.info(f"No local model found. YOLO will auto-download: {resolved_path}")

    detector = ObjectDetector(
        model_path=resolved_path,
        conf_threshold=0.35,
        iou_threshold=0.45,
        device="auto",
        tracker=TRACKER,
        models_dir=MODELS_DIR,
        imgsz=IMGSZ,
        class_subset=CLASS_FILTER,
    )
    logger.info(f"Detector initialized: {detector.model_name} on {detector.device} (imgsz={IMGSZ})")

    yield

    logger.info("Backend shut down")


app = FastAPI(
    title="Object Detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
detector: Optional[ObjectDetector] = None
current_fps = 15


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "detector_ready": detector is not None,
        "device": detector.device if detector else "unknown",
        "model": detector.model_name if detector else "unknown",
    }


@app.get("/api/stats")
async def stats():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return detector.get_stats()


@app.get("/api/classes")
async def list_classes():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return {
        "total": len(detector.classes),
        "classes": detector.classes,
        "subsets": {k: {"count": len(v), "names": [detector.classes.get(i, str(i)) for i in v[:20]]}
                    for k, v in CLASS_SUBSETS.items()},
        "active_filter": detector._class_subset_name,
    }


@app.get("/api/models")
async def list_models():
    return {
        "current": detector.model_name if detector else None,
        "available": AVAILABLE_MODELS,
        "device": detector.device if detector else "unknown",
    }


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    changes = []

    if config.conf_threshold is not None:
        old = detector.conf_threshold
        detector.update_thresholds(
            conf_threshold=max(0.01, min(0.99, config.conf_threshold)),
            iou_threshold=detector.iou_threshold,
        )
        changes.append(f"conf_threshold: {old} -> {detector.conf_threshold}")

    if config.iou_threshold is not None:
        old = detector.iou_threshold
        detector.update_thresholds(
            conf_threshold=detector.conf_threshold,
            iou_threshold=max(0.01, min(0.99, config.iou_threshold)),
        )
        changes.append(f"iou_threshold: {old} -> {detector.iou_threshold}")

    if config.max_fps is not None:
        global current_fps
        current_fps = max(1, min(60, config.max_fps))
        changes.append(f"max_fps -> {current_fps}")

    if config.tracker is not None:
        if config.tracker in ("botsort", "bytetrack"):
            detector.set_tracker(config.tracker)
            changes.append(f"tracker -> {config.tracker}")

    if config.model is not None:
        if detector.switch_model(config.model):
            changes.append(f"model -> {config.model}")
        else:
            changes.append(f"model: failed to switch to '{config.model}'")

    if config.imgsz is not None:
        valid = [320, 416, 512, 640, 800, 960, 1024, 1280]
        if config.imgsz in valid:
            detector.imgsz = config.imgsz
            changes.append(f"imgsz -> {config.imgsz}")

    if config.class_filter is not None:
        detector.set_class_filter(config.class_filter)
        changes.append(f"class_filter -> {config.class_filter}")

    return {
        "status": "ok",
        "changes": changes,
        "config": {
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "max_fps": current_fps,
            "model": detector.model_name,
            "imgsz": detector.imgsz,
            "class_filter": detector._class_subset_name,
            "tracker": detector.tracker_name,
        },
    }


@app.post("/api/model")
async def switch_model(body: Dict[str, str]):
    """Hot-swap the YOLO model at runtime."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    model_name = body.get("model", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    if detector.switch_model(model_name):
        return {"status": "ok", "model": model_name, "info": AVAILABLE_MODELS.get(model_name, {})}
    raise HTTPException(status_code=400, detail=f"Failed to switch to model '{model_name}'")


@app.post("/api/reset")
async def reset_counts():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    detector.reset_session_counts()
    return {"status": "ok", "message": "Session counts reset"}


@app.post("/api/detect")
async def detect_frame(request: Request):
    """
    Accept a base64 JPEG frame from the browser camera, run detection,
    return detection metadata.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    if not _rate_limiter.allow():
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 30 requests/sec.")

    body = await request.json()
    image_data = body.get("image")
    if not image_data:
        raise HTTPException(status_code=400, detail="No image provided")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    if len(image_data) > 10_000_000:
        raise HTTPException(status_code=400, detail="Image too large (max 7.5MB)")

    try:
        img_bytes = base64.b64decode(image_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_frame = detector.detect(frame_rgb)

    return {
        "type": "detection",
        "capture_mode": "camera",
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": [round(b, 4) for b in d.bbox],
                "tracker_id": d.tracker_id,
                "tracking_duration_frames": d.tracking_duration_frames,
                "first_seen_timestamp": round(d.first_seen_timestamp, 3) if d.first_seen_timestamp else None,
                "zone_id": d.zone_id,
            }
            for d in det_frame.detections
        ],
        "object_counts": det_frame.object_counts,
        "session_counts": det_frame.session_counts,
        "total_objects": det_frame.total_objects,
        "active_tracks_count": det_frame.active_tracks_count,
        "total_unique_seen": det_frame.total_unique_seen,
        "fps": round(det_frame.fps, 1),
        "inference_ms": round(det_frame.inference_ms, 1),
        "capture_ms": 0,
        "frame_width": det_frame.frame_width,
        "frame_height": det_frame.frame_height,
        "model": det_frame.model_name,
    }


# ---------------------------------------------------------------------------
# History / Export
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history(limit: int = 100):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return {"history": detector.get_history(limit=limit)}


@app.get("/api/export/csv")
async def export_csv():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    csv_data = detector.export_history_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=detection_{int(time.time())}.csv"},
    )


# ---------------------------------------------------------------------------
# Zone management
# ---------------------------------------------------------------------------
@app.post("/api/zones")
async def add_zone(body: Dict):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    name = body.get("name", "Zone")
    points = body.get("points", [])
    if not points or len(points) < 3:
        raise HTTPException(status_code=400, detail="Zone needs at least 3 points")
    zone = detector.add_zone(name, points)
    return {"status": "ok", "zone": {"id": zone.id, "name": zone.name, "points": zone.points}}


@app.delete("/api/zones/{zone_id}")
async def remove_zone(zone_id: int):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    if detector.remove_zone(zone_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Zone not found")


@app.post("/api/zones/clear")
async def clear_zones():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    detector.clear_zones()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    logger.warning(f"Frontend dist not found at {FRONTEND_DIST} — serving API only")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_PATH or 'yolov8l.pt (auto)'}")

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
