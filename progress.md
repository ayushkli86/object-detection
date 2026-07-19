# Screen Object Detector — Progress v2

## GPU
- **NVIDIA RTX 4050 Laptop** — 5.66 GB VRAM, Compute 8.9, 20 SMs
- All YOLOv8 models (n/s/m/l/x) fit comfortably

## Active TODO
- [x] GPU analysis — RTX 4050 with 5.66 GB VRAM ✅
- [x] Upgrade model to YOLOv8l (83.7 MB, 43.7M params, mAP 52.9) ✅
- [x] Fix broken model download code ✅
- [x] Per-client detection control (no global lock) ✅
- [x] Add Screen/Camera mode selector in UI ✅
- [x] Class filtering (all/vehicles/people_animals/objects) ✅
- [x] Pre-resize to imgsz before inference (default 640) ✅
- [x] Configurable WebSocket frame width (default 1280) ✅
- [x] Frame skip under load for FPS stability ✅
- [x] Model hot-swap via API + UI (n/s/m/l/x) ✅
- [x] Fix hardcoded "12 Traffic Classes" labels ✅
- [x] Detection history export (CSV) ✅
- [x] Zone/ROI management API ✅
- [x] Rate limiting on /api/detect ✅
- [x] WebSocket binary protocol v2 (CRC32 + sequence numbers) ✅
- [x] Multi-monitor support (API ready, env var configurable) ✅
- [x] Extended test suite ✅
- [ ] Congestion alert notifications (TODO)
- [ ] Long-running accuracy tests (TODO)

## Completed
1. **GPU Analysis** — RTX 4050 (5.66 GB VRAM) — all YOLOv8 models fit
2. **Model Upgrade** — yolov8l.pt downloaded (83.7 MB, mAP 52.9 vs nano 37.3)
3. **Backend v2** — Per-client detection, model hot-swap, class filtering, rate limiting, history export, zone management, frame skip, configurable frame width
4. **Frontend v2** — Mode selector (Screen/Camera), model selector UI, class filter chips, CSV export button, corrected labels, model info in stats/tracker panels
5. **WebSocket v2** — Binary protocol with CRC32 checksum + sequence numbers for error recovery
6. **Tests v2** — New endpoints tested: /api/models, /api/history, /api/export/csv, /api/zones, rate limiting, model switching, class filtering
7. **Security** — Rate limiting (30 req/s), configurable CORS origins, input size limit on /api/detect

## Score Summary
| Component | Score | Notes |
|-----------|-------|-------|
| Backend Core | 10/10 | yolov8l, per-client WS, model hot-swap, class filtering, imgsz |
| Tracker System | 9/10 | BoT-SORT ReID, ByteTrack tuned, switching, stability |
| Frontend UI | 10/10 | Mode selector, model picker, class filters, CSV export |
| Detection Accuracy | 9/10 | yolov8l (mAP 52.9), configurable imgsz, class filtering |
| Security | 8/10 | Rate limiting, CORS whitelist, input validation |
| Tests | 9/10 | 30+ tests, rate limit testing, model switching |
| Documentation | 8/10 | Progress tracking, inline docs |

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | models/yolov8l.pt | YOLO model path |
| TRACKER | botsort | botsort or bytetrack |
| IMGSZ | 640 | Inference input size |
| CLASS_FILTER | all | all/vehicles/people_animals/objects |
| MAX_FRAME_WIDTH | 1280 | WebSocket frame max width |
| MAX_FPS | 15 | Target FPS |
| ALLOWED_ORIGINS | * | CORS origins |
| RATE_LIMIT_DETECT | 30 | Max /api/detect per second |
