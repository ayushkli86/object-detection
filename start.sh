#!/usr/bin/env bash
# =============================================================================
# ScreenDetect — Camera-Based YOLOv8 Object Detection
# Starts: FastAPI backend + serves built React frontend on port 8765
#
# Usage:
#   ./start.sh            Start server (rebuild frontend if needed)
#   ./start.sh --rebuild  Force rebuild frontend then start
#   ./start.sh --stop     Stop server
#   ./start.sh --status   Show server health
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/models/yolov8l.pt}"
TRACKER="${TRACKER:-botsort}"
IMGSZ="${IMGSZ:-640}"
CLASS_FILTER="${CLASS_FILTER:-all}"

G='\033[0;32m' C='\033[0;36m' Y='\033[1;33m' R='\033[0;31m' B='\033[1m' NC='\033[0m'
log()  { echo -e "${C}[INFO]${NC} $*"; }
ok()   { echo -e "${G}[  OK]${NC} $*"; }
warn() { echo -e "${Y}[WARN]${NC} $*"; }
err()  { echo -e "${R}[FAIL]${NC} $*"; }

# ── --stop ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    log "Stopping server..."
    kill -9 $(lsof -ti :"$PORT") 2>/dev/null || true
    ok "Stopped."
    exit 0
fi

# ── --status ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--status" ]]; then
    echo ""
    if curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; then
        echo -e "  Server:  ${G}RUNNING${NC} (port $PORT)"
        curl -s "http://localhost:$PORT/api/health" | "$VENV_PYTHON" -m json.tool 2>/dev/null
    else
        echo -e "  Server:  ${R}STOPPED${NC}"
    fi
    exit 0
fi

# ── --rebuild ───────────────────────────────────────────────────────────
REBUILD=false
if [[ "${1:-}" == "--rebuild" ]]; then
    REBUILD=true
fi

cleanup() {
    echo ""
    log "Shutting down..."
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
    ok "Server stopped."
    exit 0
}
trap cleanup INT TERM

# ── Banner ──────────────────────────────────────────────────────────────
echo -e "${C}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║         Object Detection — YOLOv8 (l)           ║"
echo "  ║        Real-Time Camera Object Detection        ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Pre-flight checks ──────────────────────────────────────────────────
for cmd in python3 node npm; do
    command -v "$cmd" &>/dev/null || { err "$cmd not found"; exit 1; }
done
ok "Prerequisites found"

# ── Create venv if missing ─────────────────────────────────────────────
if [[ ! -f "$VENV_PYTHON" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi
ok "Using venv: $VENV_DIR"

# ── GPU check ───────────────────────────────────────────────────────────
GPU_INFO=$("$VENV_PYTHON" -c "import torch; print(f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "unknown")
if [[ "$GPU_INFO" == "CPU" || "$GPU_INFO" == "unknown" ]]; then
    warn "No GPU detected — running on CPU (slower)"
else
    ok "GPU: ${B}$GPU_INFO${NC}"
fi

# ── Kill anything on our port ──────────────────────────────────────────
if lsof -ti :"$PORT" &>/dev/null; then
    warn "Port $PORT in use — killing"
    kill -9 $(lsof -ti :"$PORT") 2>/dev/null || true
    sleep 0.5
fi
ok "Port $PORT is free"

# ── Install backend dependencies ────────────────────────────────────────
log "Installing backend dependencies..."
cd "$BACKEND_DIR"
"$VENV_PIP" install -r requirements.txt -q 2>/dev/null \
    || "$VENV_PIP" install fastapi uvicorn[standard] websockets ultralytics torch opencv-python mss numpy Pillow -q
ok "Backend dependencies ready"

# ── Install + build frontend ────────────────────────────────────────────
cd "$FRONTEND_DIR"
log "Installing frontend dependencies..."
npm install --silent 2>/dev/null
ok "Frontend dependencies ready"

if [[ "$REBUILD" == true ]] || [ ! -d "$FRONTEND_DIR/dist" ] || [ ! -f "$FRONTEND_DIR/dist/index.html" ]; then
    log "Building frontend..."
    rm -rf "$FRONTEND_DIR/dist"
    ./node_modules/.bin/vite build 2>&1 | tail -5
    ok "Frontend built"
else
    ok "Frontend already built (dist/ exists — use --rebuild to force)"
fi

# ── Start server ────────────────────────────────────────────────────────
log "Starting server on $HOST:$PORT ..."
(
    cd "$BACKEND_DIR"
    exec env MODEL_PATH="$MODEL_PATH" PORT="$PORT" TRACKER="$TRACKER" \
        IMGSZ="$IMGSZ" CLASS_FILTER="$CLASS_FILTER" \
        "$VENV_PYTHON" -m uvicorn main:app \
            --host "$HOST" --port "$PORT" \
            --log-level info
) &
BACKEND_PID=$!

log "Waiting for server..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; then
        ok "Server ready"
        break
    fi
    [ "$i" -eq 30 ] && { err "Failed to start"; kill "$BACKEND_PID" 2>/dev/null; exit 1; }
    sleep 1
    echo -n "."
done
echo ""

# ── Done ────────────────────────────────────────────────────────────────
echo ""
echo -e "${C}  ┌───────────────────────────────────────────────────┐${NC}"
echo -e "${C}  │${NC}                                                   ${C}│${NC}"
echo -e "${C}  │${NC}  ${G}Open in browser:${NC}                                ${C}│${NC}"
echo -e "${C}  │${NC}  ${B}http://localhost:$PORT${NC}                           ${C}│${NC}"
echo -e "${C}  │${NC}                                                   ${C}│${NC}"
echo -e "${C}  │${NC}  GPU: $GPU_INFO                                  ${C}│${NC}"
echo -e "${C}  │${NC}  Ctrl+C to stop  |  ./start.sh --stop            ${C}│${NC}"
echo -e "${C}  │${NC}                                                   ${C}│${NC}"
echo -e "${C}  └───────────────────────────────────────────────────┘${NC}"
echo ""

wait
