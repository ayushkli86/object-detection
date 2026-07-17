"""
Object Detection - Backend Server
FastAPI + WebSocket server for real-time YOLOv8 screen/camera object detection.

Architecture:
  - WebSocket `/ws` streams detection frames (JPEG + detection data)
  - REST API for configuration, stats, and control
  - Multiprocessing engine for non-blocking detection
"""

import os
import sys
import json
import asyncio
import logging
import struct
import time
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

# Ensure backend/ is on sys.path so local imports resolve from any CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import ObjectDetector
from screen_capture import ScreenCapture, CaptureRegion

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("screen-detector")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/yolov8n.pt"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
MAX_FPS = int(os.environ.get("MAX_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
TRACKER = os.environ.get("TRACKER", "botsort")  # "botsort" or "bytetrack"
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
FRAME_INTERVAL = 1.0 / MAX_FPS

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
detector: Optional[ObjectDetector] = None
screen_capture: Optional[ScreenCapture] = None
capture_mode: str = "screen"  # "screen" or "camera"
detection_active = False
current_fps = MAX_FPS
stream_clients: set = set()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ConfigUpdate(BaseModel):
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    max_fps: Optional[int] = None
    monitor_id: Optional[int] = None
    capture_region: Optional[Dict[str, int]] = None
    capture_mode: Optional[str] = None  # "screen" or "camera"
    tracker: Optional[str] = None       # "botsort" or "bytetrack"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, screen_capture
    logger.info("Starting Screen Object Detector backend...")

    # Initialize detector
    resolved_path = os.path.expanduser(MODEL_PATH)
    if not os.path.exists(resolved_path):
        # Try downloading
        logger.warning(f"Model not found at {resolved_path}, attempting download...")
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model.save(resolved_path)
        logger.info(f"Model downloaded to {resolved_path}")

    detector = ObjectDetector(
        model_path=resolved_path,
        conf_threshold=0.35,
        iou_threshold=0.45,
        device="auto",
        tracker=TRACKER,
        models_dir=MODELS_DIR,
    )
    logger.info(f"Detector initialized on {detector.device}")

    # Initialize screen capture
    try:
        screen_capture = ScreenCapture()
        monitors = screen_capture.get_monitors()
        logger.info(f"Screen capture initialized: {len(monitors)} monitor(s)")
        for m in monitors:
            logger.info(f"  Monitor {m.id}: {m.width}x{m.height} @ ({m.left},{m.top})")
    except Exception as e:
        logger.error(f"Failed to initialize screen capture: {e}")
        screen_capture = None

    # NOTE: Camera is captured from the browser (getUserMedia), not the backend.
    # The backend only receives base64 frames via POST /api/detect.
    # Do NOT open the camera here — it would lock the device from the browser.

    yield

    # Shutdown
    global detection_active
    detection_active = False
    if screen_capture:
        screen_capture.cleanup()
    logger.info("Backend shut down")


app = FastAPI(
    title="Object Detection",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend from any origin in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Binary WebSocket protocol:
#   [4 bytes: payload length][payload]
#   payload = JSON metadata (4 bytes length + UTF-8) + JPEG bytes
# ---------------------------------------------------------------------------
def encode_frame(frame_bgr: np.ndarray, detections_data: Dict) -> bytes:
    """Encode annotated frame + detection metadata into binary message."""
    # Resize for faster WebSocket transmission (max 800px wide)
    h, w = frame_bgr.shape[:2]
    if w > 800:
        scale = 800 / w
        frame_bgr = cv2.resize(frame_bgr, (800, int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # Encode frame as JPEG
    success, jpeg_data = cv2.imencode(".jpg", frame_bgr, [
        cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY
    ])
    if not success:
        return b""

    jpeg_bytes = jpeg_data.tobytes()

    # Serialize metadata
    meta_json = json.dumps(detections_data).encode("utf-8")

    # Build binary message:
    # [4 bytes: total_length][4 bytes: meta_length][meta_json][jpeg_bytes]
    meta_len = len(meta_json)
    total_len = 4 + meta_len + len(jpeg_bytes)

    msg = struct.pack("!II", total_len, meta_len) + meta_json + jpeg_bytes
    return msg


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "detector_active": detection_active,
        "detector_ready": detector is not None,
        "screen_capture_ready": screen_capture is not None,
        "capture_mode": capture_mode,
        "fps": current_fps,
        "clients": len(stream_clients),
        "device": detector.device if detector else "unknown",
    }


@app.get("/api/stats")
async def stats():
    """Get detection statistics."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return detector.get_stats()


@app.get("/api/monitors")
async def list_monitors():
    """List available monitors."""
    if screen_capture is None:
        raise HTTPException(status_code=503, detail="Screen capture not initialized")
    return [vars(m) for m in screen_capture.get_monitors()]


@app.get("/api/classes")
async def list_classes():
    """List all detectable COCO classes."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    return {
        "total": len(detector.classes),
        "classes": detector.classes,
    }


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Update detection configuration."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    changes = []

    if config.conf_threshold is not None:
        old_conf = detector.conf_threshold
        detector.update_thresholds(
            conf_threshold=max(0.01, min(0.99, config.conf_threshold)),
            iou_threshold=detector.iou_threshold,
        )
        changes.append(f"conf_threshold: {old_conf} -> {detector.conf_threshold}")

    if config.iou_threshold is not None:
        detector.update_thresholds(
            conf_threshold=detector.conf_threshold,
            iou_threshold=max(0.01, min(0.99, config.iou_threshold)),
        )
        changes.append(f"iou_threshold -> {detector.iou_threshold}")

    if config.max_fps is not None:
        global current_fps, FRAME_INTERVAL
        current_fps = max(1, min(60, config.max_fps))
        FRAME_INTERVAL = 1.0 / current_fps
        changes.append(f"max_fps -> {current_fps}")

    if config.monitor_id is not None and screen_capture is not None:
        screen_capture.set_monitor(config.monitor_id)
        changes.append(f"monitor -> {config.monitor_id}")

    if config.capture_region is not None and screen_capture is not None:
        region = CaptureRegion(**config.capture_region)
        screen_capture.set_region(region)
        changes.append(f"region -> {config.capture_region}")

    if config.capture_mode is not None:
        global capture_mode
        if config.capture_mode in ("screen", "camera"):
            capture_mode = config.capture_mode
            changes.append(f"capture_mode -> {capture_mode}")

    if config.tracker is not None:
        if config.tracker in ("botsort", "bytetrack"):
            detector.set_tracker(config.tracker)
            changes.append(f"tracker -> {config.tracker}")
        else:
            changes.append(f"tracker: invalid '{config.tracker}' (use botsort or bytetrack)")

    return {
        "status": "ok",
        "changes": changes,
        "config": {
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "max_fps": current_fps,
        },
    }


@app.post("/api/reset")
async def reset_counts():
    """Reset session object counts."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    detector.reset_session_counts()
    return {"status": "ok", "message": "Session counts reset"}


@app.post("/api/screenshot")
async def take_screenshot():
    """Take a single detection screenshot."""
    if detector is None or screen_capture is None:
        raise HTTPException(status_code=503, detail="Detector or capture not initialized")

    frame, cap_ms = screen_capture.capture()
    det_frame = detector.detect(frame)

    # Annotate
    annotated = detector.draw_detections(frame, det_frame.detections)

    # Save
    os.makedirs("screenshots", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"screenshots/detection_{ts}.jpg"
    cv2.imwrite(path, annotated)

    return {
        "status": "ok",
        "path": path,
        "detections": len(det_frame.detections),
        "objects": det_frame.object_counts,
    }


@app.post("/api/detect")
async def detect_frame(request: Request):
    """
    Accept a base64 JPEG/PNG frame from browser camera, run YOLO detection,
    return annotated frame + detection data.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    body = await request.json()
    image_data = body.get("image")  # base64 data URL or raw base64
    if not image_data:
        raise HTTPException(status_code=400, detail="No image provided")

    # Decode base64 image
    import base64
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]  # strip data:image/jpeg;base64,

    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Convert BGR to RGB (YOLO expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    detect_start = time.time()
    det_frame = detector.detect(frame_rgb)
    infer_ms = (time.time() - detect_start) * 1000

    # Annotate frame
    annotated = detector.draw_detections(frame_rgb, det_frame.detections)

    # Encode annotated frame as JPEG
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    _, jpeg_buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    annotated_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

    return {
        "type": "detection",
        "capture_mode": "camera",
        "image": f"data:image/jpeg;base64,{annotated_b64}",
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": [round(b, 4) for b in d.bbox],
                "tracker_id": d.tracker_id,
            }
            for d in det_frame.detections
        ],
        "object_counts": det_frame.object_counts,
        "session_counts": det_frame.session_counts,
        "total_objects": det_frame.total_objects,
        "active_tracks_count": det_frame.active_tracks_count,
        "total_unique_seen": det_frame.total_unique_seen,
        "fps": round(det_frame.fps, 1),
        "inference_ms": round(infer_ms, 1),
        "capture_ms": 0,
        "frame_width": det_frame.frame_width,
        "frame_height": det_frame.frame_height,
    }


# ---------------------------------------------------------------------------
# WebSocket - Real-time detection stream
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global detection_active, current_fps, capture_mode
    await websocket.accept()
    stream_clients.add(websocket)
    client_addr = websocket.client
    logger.info(f"WebSocket client connected: {client_addr}")

    try:
        while True:
            # ── Drain ALL pending control messages before streaming ────────
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.05
                    )
                    cmd = json.loads(message)

                    if cmd.get("action") == "start":
                        detection_active = True
                        logger.info("Detection started by client")
                        await websocket.send_json({"type": "status", "status": "detecting"})

                    elif cmd.get("action") == "stop":
                        detection_active = False
                        logger.info("Detection stopped by client")
                        await websocket.send_json({"type": "status", "status": "stopped"})

                    elif cmd.get("action") == "config":
                        if detector:
                            if "conf" in cmd:
                                detector.update_thresholds(
                                    conf_threshold=cmd["conf"],
                                    iou_threshold=detector.iou_threshold,
                                )
                            if "iou" in cmd:
                                detector.update_thresholds(
                                    conf_threshold=detector.conf_threshold,
                                    iou_threshold=cmd["iou"],
                                )
                            await websocket.send_json({
                                "type": "config",
                                "conf_threshold": detector.conf_threshold,
                                "iou_threshold": detector.iou_threshold,
                            })

                    elif cmd.get("action") == "reset":
                        if detector:
                            detector.reset_session_counts()
                            await websocket.send_json({
                                "type": "status",
                                "status": "counts_reset",
                            })

                    elif cmd.get("action") == "set_mode":
                        new_mode = cmd.get("mode", "screen")
                        if new_mode in ("screen", "camera"):
                            capture_mode = new_mode
                            logger.info(f"Capture mode set to: {capture_mode}")
                            await websocket.send_json({
                                "type": "status",
                                "status": "mode_changed",
                                "capture_mode": capture_mode,
                            })

                    elif cmd.get("action") == "set_tracker":
                        new_tracker = cmd.get("tracker", "botsort")
                        if new_tracker in ("botsort", "bytetrack") and detector:
                            detector.set_tracker(new_tracker)
                            logger.info(f"Tracker set to: {new_tracker}")
                            await websocket.send_json({
                                "type": "status",
                                "status": "tracker_changed",
                                "tracker": new_tracker,
                            })

                except asyncio.TimeoutError:
                    break  # No more pending messages
                except json.JSONDecodeError:
                    continue  # Skip bad messages
                except WebSocketDisconnect:
                    return

            # ── Stream detection frames or send keepalive ──────────────────
            if detection_active and detector:
                detect_start = time.time()

                # Capture from screen
                if screen_capture:
                    frame, cap_ms = screen_capture.capture()
                else:
                    # No capture source available
                    await asyncio.sleep(0.1)
                    continue

                # Run detection
                det_frame = detector.detect(frame)

                # Update global FPS
                current_fps = det_frame.fps

                # Annotate frame
                annotated = detector.draw_detections(frame, det_frame.detections)

                # Build detection data
                detections_data = {
                    "type": "detection",
                    "capture_mode": capture_mode,
                    "detections": [
                        {
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "confidence": round(d.confidence, 3),
                            "bbox": [round(b, 4) for b in d.bbox],
                        }
                        for d in det_frame.detections
                    ],
                    "object_counts": det_frame.object_counts,
                    "session_counts": dict(detector.session_counts),
                    "total_objects": det_frame.total_objects,
                    "active_tracks_count": det_frame.active_tracks_count,
                    "total_unique_seen": det_frame.total_unique_seen,
                    "fps": round(det_frame.fps, 1),
                    "inference_ms": det_frame.inference_ms,
                    "capture_ms": round(cap_ms, 1),
                    "frame_width": det_frame.frame_width,
                    "frame_height": det_frame.frame_height,
                }

                # Encode and send binary frame
                try:
                    msg = encode_frame(annotated, detections_data)
                    if msg:
                        await websocket.send_bytes(msg)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(f"Send error: {e}")
                    break

                # Throttle to target FPS
                elapsed = (time.time() - detect_start) * 1000
                min_interval = FRAME_INTERVAL * 1000
                if elapsed < min_interval:
                    await asyncio.sleep((min_interval - elapsed) / 1000)
            else:
                # No detection active - send keepalive
                try:
                    await asyncio.wait_for(
                        websocket.send_json({"type": "ping"}),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    pass
                except WebSocketDisconnect:
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        stream_clients.discard(websocket)
        detection_active = False


# ---------------------------------------------------------------------------
# Serve frontend static files (built React app)
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    # Mount static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        """Serve frontend for all non-API/non-WS routes (SPA fallback)."""
        # Try serving the exact file first
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fall back to index.html for SPA routing
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    logger.warning(f"Frontend dist not found at {FRONTEND_DIST} — serving API only")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Max FPS: {MAX_FPS}")
    logger.info(f"JPEG Quality: {JPEG_QUALITY}")

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
