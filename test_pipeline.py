#!/usr/bin/env python3
"""
Camera Object Detector — Full Pipeline Test
Tests all REST API endpoints.
"""

import base64, sys
import cv2, numpy as np
import requests

BASE = "http://localhost:8765"
passed = 0
failed = 0
errors = []

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  ✓ PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  ✗ FAIL  {name}: {e}")

print("\n═══ Camera Object Detection Tests ═══\n")

# 1. Health
def t_health():
    r = requests.get(f"{BASE}/api/health", timeout=5)
    d = r.json()
    assert d["status"] == "ok"
    assert d["detector_ready"] is True
    assert "cuda" in d["device"] or "cpu" in d["device"]
run_test("GET /api/health", t_health)

# 2. Models
def t_models():
    r = requests.get(f"{BASE}/api/models", timeout=5)
    d = r.json()
    assert d["current"] is not None
    assert len(d["available"]) >= 3
    assert "besst" in d["available"]
run_test("GET /api/models", t_models)

# 3. Classes
def t_classes():
    r = requests.get(f"{BASE}/api/classes", timeout=5)
    d = r.json()
    assert d["total"] == 80
    assert "subsets" in d
run_test("GET /api/classes", t_classes)

# 4. Stats
def t_stats():
    r = requests.get(f"{BASE}/api/stats", timeout=5)
    d = r.json()
    assert "device" in d
    assert "model" in d
    assert "frames_processed" in d
run_test("GET /api/stats", t_stats)

# 5. Config
def t_config():
    r = requests.post(f"{BASE}/api/config", json={"conf_threshold": 0.5}, timeout=5)
    assert r.status_code == 200
    requests.post(f"{BASE}/api/config", json={"conf_threshold": 0.35}, timeout=5)
run_test("POST /api/config", t_config)

# 6. Detect with JPEG
def t_detect():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf.tobytes()).decode()
    r = requests.post(f"{BASE}/api/detect", json={"image": b64}, timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["type"] == "detection"
    assert d["capture_mode"] == "camera"
    assert "model" in d
    assert d["inference_ms"] > 0
run_test("POST /api/detect", t_detect)

# 7. Detect bad input
def t_detect_empty():
    r = requests.post(f"{BASE}/api/detect", json={"image": ""}, timeout=5)
    assert r.status_code == 400
run_test("POST /api/detect — empty rejected", t_detect_empty)

def t_detect_missing():
    r = requests.post(f"{BASE}/api/detect", json={}, timeout=5)
    assert r.status_code == 400
run_test("POST /api/detect — missing rejected", t_detect_missing)

# 8. Reset
def t_reset():
    r = requests.post(f"{BASE}/api/reset", timeout=5)
    assert r.status_code == 200
run_test("POST /api/reset", t_reset)

# 9. History
def t_history():
    r = requests.get(f"{BASE}/api/history?limit=5", timeout=5)
    d = r.json()
    assert "history" in d
run_test("GET /api/history", t_history)

# 10. CSV export
def t_csv():
    r = requests.get(f"{BASE}/api/export/csv", timeout=5)
    assert r.status_code == 200
    assert "csv" in r.headers.get("content-type", "")
run_test("GET /api/export/csv", t_csv)

# 11. Frontend
def t_frontend():
    r = requests.get(f"{BASE}/", timeout=5)
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text
run_test("GET / (frontend)", t_frontend)

# 12. Zones
def t_zones():
    r = requests.post(f"{BASE}/api/zones", json={
        "name": "Test", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
    }, timeout=5)
    assert r.status_code == 200
    requests.post(f"{BASE}/api/zones/clear", timeout=5)
run_test("POST /api/zones + clear", t_zones)

# 13. Rate limiting (returns cached result instead of 429 now — just verify endpoint works)
def t_rate_limit():
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf.tobytes()).decode()
    r = requests.post(f"{BASE}/api/detect", json={"image": b64}, timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
run_test("POST /api/detect (rate limit bypass)", t_rate_limit)

# Results
print(f"\n═══ {passed}/{passed + failed} passed, {failed} failed ═══\n")
sys.exit(0 if failed == 0 else 1)
