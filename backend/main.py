"""
Object Detection - Backend Server (Camera Only)
FastAPI server for real-time YOLO camera object detection.
Single endpoint: POST /api/detect — accepts base64 JPEG, returns detections.
"""

import os
import sys
import logging
import time
import asyncio
import base64
import uuid
import socket
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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
# Detect LAN IP
# ---------------------------------------------------------------------------
def get_lan_ip() -> str:
    """Get the primary LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        # Doesn't need to be reachable
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    # Fallback
    try:
        import subprocess
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2)
        ips = result.stdout.strip().split()
        if ips:
            return ips[0]
    except Exception:
        pass
    return "localhost"

LAN_IP = get_lan_ip()
logger.info(f"LAN IP detected: {LAN_IP}")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_per_sec: int):
        self.max_per_sec = max_per_sec
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            now = time.time()
            self._timestamps = [t for t in self._timestamps if now - t < 1.0]
            if len(self._timestamps) >= self.max_per_sec:
                return False
            self._timestamps.append(now)
            return True

_rate_limiter = RateLimiter(RATE_LIMIT_DETECT_PER_SEC)


# ---------------------------------------------------------------------------
# Remote Camera Source — mobile phones streaming via WebSocket
# ---------------------------------------------------------------------------
class RemoteCameraSource:
    """Tracks a mobile phone camera streaming via WebSocket."""
    def __init__(self, camera_id: str, name: str, websocket):
        self.id = camera_id
        self.name = name
        self.websocket = websocket
        self.latest_jpeg: Optional[bytes] = None
        self.latest_detection: Optional[dict] = None
        self.connected_at = time.time()
        self.last_seen = time.time()

_remote_cameras: Dict[str, RemoteCameraSource] = {}
_active_camera = "local"  # "local" or "remote:<id>"
_camera_lock = asyncio.Lock()


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
        for candidate in ["yolov8l.pt"]:
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

    if detector is not None:
        del detector.model
        import torch
        torch.cuda.empty_cache()
        logger.info("Detector resources released")
    logger.info("Backend shut down")


app = FastAPI(
    title="Traffic Object Detection System",
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
detector_lock = asyncio.Lock()


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
        "lan_ip": f"{LAN_IP}:{PORT}",
    }


@app.get("/api/ping")
async def ping():
    """Simple connectivity test — returns immediately (no DB, no model)."""
    return {"pong": True, "lan_ip": f"{LAN_IP}:{PORT}"}


@app.get("/api/stats")
async def stats():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    async with detector_lock:
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

    async with detector_lock:
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
    async with detector_lock:
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
    async with detector_lock:
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

    if not await _rate_limiter.allow():
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 30 requests/sec.")

    async with detector_lock:
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
    async with detector_lock:
        return {"history": detector.get_history(limit=limit)}


@app.get("/api/export/csv")
async def export_csv():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    async with detector_lock:
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
    if not all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in points):
        raise HTTPException(status_code=400, detail="Each point must be [x, y] with numeric values")
    async with detector_lock:
        zone = detector.add_zone(name, points)
    return {"status": "ok", "zone": {"id": zone.id, "name": zone.name, "points": zone.points}}


@app.delete("/api/zones/{zone_id}")
async def remove_zone(zone_id: int):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    async with detector_lock:
        if detector.remove_zone(zone_id):
            return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Zone not found")


@app.post("/api/zones/clear")
async def clear_zones():
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    async with detector_lock:
        detector.clear_zones()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Remote Cameras — mobile phone streaming via WebSocket
# ---------------------------------------------------------------------------

@app.get("/api/cameras")
async def list_cameras():
    """List all available camera sources (local + remotes)."""
    async with _camera_lock:
        cams = [
            {"id": "local", "name": "Local Camera", "type": "local", "active": _active_camera == "local", "last_seen": time.time()}
        ]
        for cam_id, cam in list(_remote_cameras.items()):
            # Cleanup stale cameras (>30s no frame)
            if time.time() - cam.last_seen > 30:
                continue
            cams.append({
                "id": cam_id,
                "name": cam.name,
                "type": "remote",
                "active": _active_camera == cam_id,
                "last_seen": cam.last_seen,
            })
        return {"cameras": cams}


@app.post("/api/cameras/select")
async def select_camera(body: Dict):
    """Switch the active camera source."""
    global _active_camera
    cam_id = body.get("camera_id", "local")
    async with _camera_lock:
        if cam_id != "local" and cam_id not in _remote_cameras:
            raise HTTPException(status_code=404, detail="Camera not found")
        _active_camera = cam_id
    logger.info(f"Active camera switched to: {cam_id}")
    return {"status": "ok", "selected": cam_id}


@app.get("/api/remote-frame")
async def get_remote_frame():
    """Get the latest frame + detection from the active remote camera."""
    async with _camera_lock:
        cam_id = _active_camera
    if cam_id == "local":
        raise HTTPException(status_code=400, detail="Local camera selected — use /api/detect instead")

    async with detector_lock:
        cam = _remote_cameras.get(cam_id)
        if not cam or cam.latest_jpeg is None:
            raise HTTPException(status_code=404, detail="No frame available from this camera")

        det = cam.latest_detection or {}
        jpeg_b64 = base64.b64encode(cam.latest_jpeg).decode()

    return {
        "type": "detection",
        "image": f"data:image/jpeg;base64,{jpeg_b64}",
        "detections": det.get("detections", []),
        "object_counts": det.get("object_counts", {}),
        "session_counts": det.get("session_counts", {}),
        "total_objects": det.get("total_objects", 0),
        "active_tracks_count": det.get("active_tracks_count", 0),
        "total_unique_seen": det.get("total_unique_seen", 0),
        "fps": round(det.get("fps", 0), 1),
        "inference_ms": round(det.get("inference_ms", 0), 1),
        "frame_width": det.get("frame_width", 0),
        "frame_height": det.get("frame_height", 0),
        "model": det.get("model", "yolov8l"),
    }


@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for mobile phones to stream camera frames.
    Query params: ?name=MyPhone
    Mobile sends binary JPEG frames, server runs detection and returns JSON results.
    """
    name = websocket.query_params.get("name", "Mobile Camera")
    await websocket.accept()

    if detector is None:
        await websocket.send_json({"type": "error", "message": "Detector not initialized"})
        await websocket.close()
        return

    camera_id = f"remote:{uuid.uuid4().hex[:8]}"

    cam = RemoteCameraSource(camera_id, name, websocket)
    async with _camera_lock:
        _remote_cameras[camera_id] = cam
    logger.info(f"Remote camera connected: {name} ({camera_id})")

    try:
        # Send welcome with camera ID
        await websocket.send_json({"type": "welcome", "camera_id": camera_id})

        while True:
            # Receive binary frame (raw JPEG bytes)
            data = await websocket.receive_bytes()

            async with detector_lock:
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det_frame = detector.detect(frame_rgb)

                # Store latest frame + detection
                cam.latest_jpeg = data
                cam.last_seen = time.time()
                cam.latest_detection = {
                    "type": "detection",
                    "detections": [
                        {
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "confidence": round(d.confidence, 3),
                            "bbox": [round(b, 4) for b in d.bbox],
                            "tracker_id": d.tracker_id,
                            "tracking_duration_frames": d.tracking_duration_frames,
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
                    "frame_width": det_frame.frame_width,
                    "frame_height": det_frame.frame_height,
                    "model": det_frame.model_name,
                }

            # Send detection results back to mobile
            try:
                await websocket.send_json(cam.latest_detection)
            except Exception:
                pass

    except WebSocketDisconnect:
        logger.info(f"Remote camera disconnected: {name} ({camera_id})")
    except Exception as e:
        logger.warning(f"Remote camera error ({name}): {e}")
    finally:
        async with _camera_lock:
            _remote_cameras.pop(camera_id, None)
            if _active_camera == camera_id:
                _active_camera = "local"
        logger.info(f"Remote camera removed: {name} ({camera_id})")


@app.get("/api/all-frames")
async def get_all_frames():
    """Return latest frame + detection for ALL remote cameras at once."""
    result = {"cameras": {}, "active_camera": _active_camera}
    async with _camera_lock:
        for cam_id, cam in list(_remote_cameras.items()):
            if cam.latest_jpeg is None:
                continue
            # Expire stale cameras
            if time.time() - cam.last_seen > 30:
                continue
            det = cam.latest_detection or {}
            result["cameras"][cam_id] = {
                "name": cam.name,
                "type": "remote",
                "active": cam_id == _active_camera,
                "detections": det.get("detections", []),
                "total_objects": det.get("total_objects", 0),
                "active_tracks_count": det.get("active_tracks_count", 0),
                "fps": det.get("fps", 0),
                "inference_ms": det.get("inference_ms", 0),
            }
    return result


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        # Serve mobile.html from backend directory (for phone streaming page)
        if full_path == "mobile.html":
            mobile_backend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobile.html")
            if os.path.isfile(mobile_backend):
                return FileResponse(mobile_backend)
        resolved = os.path.normpath(os.path.join(FRONTEND_DIST, full_path))
        if not resolved.startswith(os.path.normpath(FRONTEND_DIST)):
            raise HTTPException(status_code=403)
        if full_path and os.path.isfile(resolved):
            return FileResponse(resolved)
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
