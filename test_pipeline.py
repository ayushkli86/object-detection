#!/usr/bin/env python3
"""
Screen Object Detector — Full Pipeline Test
Tests EVERY functionality end-to-end.
"""

import asyncio
import json
import os
import sys
import time
import base64

import requests
import cv2
import numpy as np
import websockets

BASE = "http://localhost:8765"
WS_URL = "ws://localhost:8765/ws"

passed = 0
failed = 0
errors = []
total = 0

def run_test(name, fn):
    global passed, failed, total
    total += 1
    try:
        result = fn()
        if result is False:
            failed += 1
            errors.append(name)
            print(f"  ✗ FAIL  {name}")
        else:
            passed += 1
            print(f"  ✓ PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append(f"{name}: {e}")
        print(f"  ✗ FAIL  {name}: {e}")


# ════════════════════════════════════════════════════════════════════════
# SECTION 1: REST APIs
# ════════════════════════════════════════════════════════════════════════
print("\n═══ 1. REST APIs ═══")

def t_health():
    r = requests.get(f"{BASE}/api/health", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["detector_ready"] is True
    assert d["screen_capture_ready"] is True
    assert "device" in d
run_test("GET /api/health", t_health)

def t_health_gpu():
    r = requests.get(f"{BASE}/api/health", timeout=5)
    d = r.json()
    assert "cuda" in d["device"], f"Expected CUDA, got {d['device']}"
run_test("GET /api/health — GPU device", t_health_gpu)

def t_stats():
    r = requests.get(f"{BASE}/api/stats", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "fps" in d
    assert "frames_processed" in d
    assert "uptime_seconds" in d
    assert "device" in d
run_test("GET /api/stats", t_stats)

def t_monitors():
    r = requests.get(f"{BASE}/api/monitors", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d, list)
    assert len(d) > 0, "No monitors found"
    assert "width" in d[0] and "height" in d[0]
run_test("GET /api/monitors", t_monitors)

def t_cameras():
    r = requests.get(f"{BASE}/api/cameras", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert "cameras" in d
run_test("GET /api/cameras", t_cameras)

def t_classes():
    r = requests.get(f"{BASE}/api/classes", timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 80, f"Expected 80 COCO classes, got {d['total']}"
    classes = d["classes"]
    # Classes are keyed by string ID or name
    class_names = list(classes.values()) if isinstance(list(classes.values())[0], str) else list(classes.keys())
    assert "person" in class_names or "person" in classes
    assert "car" in class_names or "car" in classes
run_test("GET /api/classes (80 COCO)", t_classes)

def t_config_set():
    r = requests.post(f"{BASE}/api/config", json={"conf_threshold": 0.5}, timeout=5)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert len(d["changes"]) > 0
run_test("POST /api/config — set confidence", t_config_set)

def t_config_restore():
    r = requests.post(f"{BASE}/api/config", json={"conf_threshold": 0.35}, timeout=5)
    assert r.status_code == 200
run_test("POST /api/config — restore confidence", t_config_restore)

def t_screenshot():
    r = requests.post(f"{BASE}/api/screenshot", timeout=15)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    # Screenshot saved relative to backend dir
    path = d["path"]
    assert os.path.exists(path) or os.path.exists(os.path.join("backend", path)), \
        f"Screenshot not found: {path}"
run_test("POST /api/screenshot", t_screenshot)

def t_frontend():
    r = requests.get(f"{BASE}/", timeout=5)
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text
    assert "index-" in r.text  # has built JS
run_test("Frontend served from backend", t_frontend)


# ════════════════════════════════════════════════════════════════════════
# SECTION 2: WebSocket — Screen Mode
# ════════════════════════════════════════════════════════════════════════
print("\n═══ 2. WebSocket — Screen Mode ═══")

def t_ws_connect():
    async def t():
        async with websockets.connect(WS_URL) as ws:
            # Just verify we can send/receive
            await ws.send(json.dumps({"action": "reset"}))
            msg = await asyncio.wait_for(ws.recv(), timeout=3)
            return True
    return asyncio.run(t())
run_test("WS connect", t_ws_connect)

def t_ws_start_detect():
    async def t():
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"action": "start"}))
            for _ in range(15):
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(msg, (bytes, bytearray)):
                    view = memoryview(msg)
                    meta_len = int.from_bytes(view[4:8], "big")
                    meta = json.loads(msg[8:8+meta_len])
                    assert meta["type"] == "detection"
                    assert meta["capture_mode"] == "screen"
                    assert meta["frame_width"] > 0
                    assert meta["inference_ms"] > 0
                    return True
            return False
    return asyncio.run(t())
run_test("WS start detection → screen frames", t_ws_start_detect)

def t_ws_stop():
    async def t():
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"action": "start"}))
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, (bytes, bytearray)):
                    break
            await ws.send(json.dumps({"action": "stop"}))
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("status") == "stopped":
                        return True
            return False
    return asyncio.run(t())
run_test("WS stop detection", t_ws_stop)

def t_ws_config():
    async def t():
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"action": "config", "conf": 0.6}))
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("type") == "config":
                        assert d["conf_threshold"] == 0.6
                        await ws.send(json.dumps({"action": "config", "conf": 0.35}))
                        return True
            return False
    return asyncio.run(t())
run_test("WS config update", t_ws_config)

def t_ws_reset():
    async def t():
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"action": "reset"}))
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("status") == "counts_reset":
                        return True
            return False
    return asyncio.run(t())
run_test("WS reset counts", t_ws_reset)

def t_ws_mode_switch():
    async def t():
        async with websockets.connect(WS_URL) as ws:
            # → camera
            await ws.send(json.dumps({"action": "set_mode", "mode": "camera"}))
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("status") == "mode_changed" and d["capture_mode"] == "camera":
                        # → screen
                        await ws.send(json.dumps({"action": "set_mode", "mode": "screen"}))
                        for _ in range(10):
                            msg2 = await asyncio.wait_for(ws.recv(), timeout=3)
                            if isinstance(msg2, str):
                                d2 = json.loads(msg2)
                                if d2.get("status") == "mode_changed" and d2["capture_mode"] == "screen":
                                    return True
            return False
    return asyncio.run(t())
run_test("WS mode switch: screen ↔ camera", t_ws_mode_switch)


# ════════════════════════════════════════════════════════════════════════
# SECTION 3: Camera Detection (/api/detect)
# ════════════════════════════════════════════════════════════════════════
print("\n═══ 3. Camera Detection (/api/detect) ═══")

def t_detect_jpeg():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    r = requests.post(f"{BASE}/api/detect", json={"image": b64}, timeout=15)
    assert r.status_code == 200
    d = r.json()
    assert d["type"] == "detection"
    assert d["capture_mode"] == "camera"
    assert d["image"].startswith("data:image/jpeg;base64,")
run_test("POST /api/detect — valid JPEG", t_detect_jpeg)

def t_detect_annotated():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf.tobytes()).decode()
    r = requests.post(f"{BASE}/api/detect", json={"image": b64}, timeout=15)
    d = r.json()
    img_b64 = d["image"].split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, "Could not decode annotated image"
    assert img.shape[0] > 0
run_test("POST /api/detect — annotated image decodable", t_detect_annotated)

def t_detect_fields():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf.tobytes()).decode()
    r = requests.post(f"{BASE}/api/detect", json={"image": b64}, timeout=15)
    d = r.json()
    for key in ["detections", "object_counts", "session_counts",
                "total_objects", "inference_ms", "frame_width", "frame_height"]:
        assert key in d, f"Missing key: {key}"
    assert d["inference_ms"] > 0
run_test("POST /api/detect — all fields present", t_detect_fields)

def t_detect_empty():
    r = requests.post(f"{BASE}/api/detect", json={"image": ""}, timeout=5)
    assert r.status_code == 400
run_test("POST /api/detect — empty rejected", t_detect_empty)

def t_detect_missing():
    r = requests.post(f"{BASE}/api/detect", json={}, timeout=5)
    assert r.status_code == 400
run_test("POST /api/detect — missing rejected", t_detect_missing)

def t_detect_invalid():
    r = requests.post(f"{BASE}/api/detect", json={"image": "not_base64!!!"}, timeout=5)
    assert r.status_code in (400, 500)
run_test("POST /api/detect — invalid rejected", t_detect_invalid)


# ════════════════════════════════════════════════════════════════════════
# SECTION 4: Full Camera Flow
# ════════════════════════════════════════════════════════════════════════
print("\n═══ 4. Full Camera Flow (WS + /api/detect) ═══")

def t_camera_flow():
    async def t():
        async with websockets.connect(WS_URL) as ws:
            # Switch to camera
            await ws.send(json.dumps({"action": "set_mode", "mode": "camera"}))
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("status") == "mode_changed":
                        break

            # Start
            await ws.send(json.dumps({"action": "start"}))
            for _ in range(5):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("status") == "detecting":
                        break

            # Send 3 camera frames
            for i in range(3):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                _, buf = cv2.imencode(".jpg", frame)
                b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
                r = requests.post(f"{BASE}/api/detect", json={"image": b64}, timeout=10)
                assert r.status_code == 200
                d = r.json()
                assert d["type"] == "detection"
                assert d["capture_mode"] == "camera"

            # Stop + reset to screen
            await ws.send(json.dumps({"action": "stop"}))
            for _ in range(10):
                msg = await asyncio.wait_for(ws.recv(), timeout=3)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("status") == "stopped":
                        await ws.send(json.dumps({"action": "set_mode", "mode": "screen"}))
                        return True
            return False
    return asyncio.run(t())
run_test("Camera: switch → start → 3 frames → stop", t_camera_flow)


# ════════════════════════════════════════════════════════════════════════
# SECTION 5: Concurrent Clients
# ════════════════════════════════════════════════════════════════════════
print("\n═══ 5. Concurrent Clients ═══")

def t_concurrent():
    async def t():
        async with websockets.connect(WS_URL) as ws1:
            async with websockets.connect(WS_URL) as ws2:
                await ws1.send(json.dumps({"action": "start"}))
                await ws2.send(json.dumps({"action": "start"}))
                got1 = got2 = False
                for _ in range(30):
                    try:
                        msg1 = await asyncio.wait_for(ws1.recv(), timeout=1)
                        if isinstance(msg1, (bytes, bytearray)):
                            got1 = True
                    except asyncio.TimeoutError:
                        pass
                    try:
                        msg2 = await asyncio.wait_for(ws2.recv(), timeout=1)
                        if isinstance(msg2, (bytes, bytearray)):
                            got2 = True
                    except asyncio.TimeoutError:
                        pass
                    if got1 and got2:
                        break
                await ws1.send(json.dumps({"action": "stop"}))
                await ws2.send(json.dumps({"action": "stop"}))
                return got1 and got2
    return asyncio.run(t())
run_test("Two clients get screen frames", t_concurrent)


# ════════════════════════════════════════════════════════════════════════
# SECTION 6: Performance
# ════════════════════════════════════════════════════════════════════════
print("\n═══ 6. Performance ═══")

def t_screen_fps():
    async def t():
        # Wait for any previous connections to close
        await asyncio.sleep(1)
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"action": "start"}))
            frames = 0
            t0 = None
            deadline = time.time() + 6
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    if isinstance(msg, (bytes, bytearray)):
                        if t0 is None:
                            t0 = time.time()  # Count from first frame
                        frames += 1
                except asyncio.TimeoutError:
                    pass
            await ws.send(json.dumps({"action": "stop"}))
            elapsed = time.time() - t0 if t0 else 10
            fps = frames / max(elapsed, 0.1)
            return fps > 5
    return asyncio.run(t())
run_test("Screen detection > 5 FPS", t_screen_fps)

def t_detect_latency():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf.tobytes()).decode()
    times = []
    for _ in range(5):
        t0 = time.time()
        requests.post(f"{BASE}/api/detect", json={"image": b64}, timeout=10)
        times.append((time.time() - t0) * 1000)
    avg = sum(times[2:]) / max(len(times[2:]), 1)  # skip warmup
    return avg < 500
run_test("Camera detect < 500ms (avg, skip warmup)", t_detect_latency)


# ════════════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(f"\n  RESULTS: {passed}/{total} passed, {failed}/{total} failed\n")
if errors:
    print("  FAILURES:")
    for e in errors:
        print(f"    ✗ {e}")
print("\n" + "═" * 60)
sys.exit(0 if failed == 0 else 1)
