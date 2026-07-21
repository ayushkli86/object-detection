import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { DetectionData, Detection } from '../types';

interface Props {
  videoElement: HTMLVideoElement | null;
  frameCanvas: HTMLCanvasElement | null;
  detecting: boolean;
  detectionData: DetectionData | null;
  remoteFrameUrl?: string | null;
}

const CATEGORY_COLORS: Record<string, string> = {
  car: '#3b82f6', truck: '#2563eb', bus: '#14b8a6',
  motorcycle: '#8b5cf6', bicycle: '#a78bfa',
  person: '#22c55e', cat: '#f59e0b', dog: '#d97706',
  bird: '#14b8a6', horse: '#f97316', sheep: '#c4b89a', cow: '#b4a078',
  'traffic light': '#ef4444', 'stop sign': '#dc2626', 'fire hydrant': '#ef4444',
  laptop: '#8b5cf6', 'cell phone': '#8b5cf6', tv: '#3b82f6',
  keyboard: '#94a3b8', mouse: '#8a7a6a',
  train: '#14b8a6', boat: '#14b8a6',
};
const FALLBACK_COLORS = [
  '#38b8eb', '#56cf63', '#f79646', '#c850c0',
  '#8282e6', '#ff8282', '#82ff82', '#ffc882', '#c8c8c8', '#ffff82',
];

function getColor(name: string, id: number): string {
  return CATEGORY_COLORS[name] || FALLBACK_COLORS[(id || 0) % FALLBACK_COLORS.length] || '#888888';
}

function getContrastColor(hex: string): string {
  if (!hex || typeof hex !== 'string' || hex.length < 7) return '#ffffff';
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  if (isNaN(r) || isNaN(g) || isNaN(b)) return '#ffffff';
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.5 ? '#000000' : '#ffffff';
}

const DetectionView: React.FC<Props> = ({
  videoElement, frameCanvas, detecting, detectionData, remoteFrameUrl,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const remoteImgRef = useRef<HTMLImageElement | null>(null);
  const prevRemoteUrlRef = useRef<string | null>(null);
  const remoteUrlRef = useRef<string | null>(null);

  useEffect(() => { remoteUrlRef.current = remoteFrameUrl ?? null; }, [remoteFrameUrl]);

  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [selectedDet, setSelectedDet] = useState<Detection | null>(null);

  const dataRef = useRef<DetectionData | null>(null);
  const hoveredRef = useRef<number | null>(null);
  const selectedRef = useRef<number | null>(null);
  const boxesRef = useRef<{ x1: number; y1: number; x2: number; y2: number; det: Detection }[]>([]);

  useEffect(() => { hoveredRef.current = hoveredIdx; }, [hoveredIdx]);
  useEffect(() => { selectedRef.current = selectedIdx; }, [selectedIdx]);
  useEffect(() => { dataRef.current = detectionData; }, [detectionData]);

  useEffect(() => {
    if (selectedIdx !== null && boxesRef.current[selectedIdx]) {
      setSelectedDet(boxesRef.current[selectedIdx].det);
    } else {
      setSelectedDet(null);
    }
  }, [selectedIdx]);

  // ── Load remote camera JPEG image ──────────────────────────────────
  useEffect(() => {
    if (!remoteFrameUrl) return;
    if (remoteFrameUrl === prevRemoteUrlRef.current) return;
    if (!remoteImgRef.current) remoteImgRef.current = new Image();
    remoteImgRef.current.src = remoteFrameUrl;
    prevRemoteUrlRef.current = remoteFrameUrl;
  }, [remoteFrameUrl]);

  // ── Main render loop ───────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let running = true;

    const render = () => {
      if (!running) return;
      try {

      const container = canvas.parentElement;
      if (!container) { rafRef.current = requestAnimationFrame(render); return; }

      const containerW = container.clientWidth;
      const containerH = container.clientHeight;
      if (containerW === 0 || containerH === 0) { rafRef.current = requestAnimationFrame(render); return; }

      const dpr = window.devicePixelRatio || 1;

      // Determine source to draw
      let srcW = 0, srcH = 0;
      let drawSource: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement | null = null;

      if (remoteUrlRef.current && detecting && remoteImgRef.current?.complete && remoteImgRef.current.naturalWidth > 0) {
        // Remote camera source
        const img = remoteImgRef.current;
        srcW = img.naturalWidth;
        srcH = img.naturalHeight;
        drawSource = img;
      } else if (detecting && frameCanvas) {
        // Detection active — draw from capture canvas (has latest frame)
        if (frameCanvas.width === 0 || frameCanvas.height === 0) {
          rafRef.current = requestAnimationFrame(render);
          return;
        }
        srcW = frameCanvas.width;
        srcH = frameCanvas.height;
        drawSource = frameCanvas;
      } else if (!detecting && videoElement) {
        // Idle — draw directly from video element for live preview
        if (videoElement.readyState < 2) {
          rafRef.current = requestAnimationFrame(render);
          return;
        }
        srcW = videoElement.videoWidth || 640;
        srcH = videoElement.videoHeight || 480;
        drawSource = videoElement;
      } else {
        rafRef.current = requestAnimationFrame(render);
        return;
      }

      // Fill container (cover mode) — video fills the entire frame
      const srcAspect = srcW / srcH;
      const containerAspect = containerW / containerH;
      let drawW: number, drawH: number;
      if (srcAspect > containerAspect) {
        // Source is wider — match height, width overflows (cropped left/right)
        drawH = containerH;
        drawW = containerH * srcAspect;
      } else {
        // Source is taller — match width, height overflows (cropped top/bottom)
        drawW = containerW;
        drawH = containerW / srcAspect;
      }

      const pixW = Math.round(drawW * dpr);
      const pixH = Math.round(drawH * dpr);
      if (pixW !== canvas.width || pixH !== canvas.height) {
        canvas.width = pixW;
        canvas.height = pixH;
        canvas.style.width = `${drawW}px`;
        canvas.style.height = `${drawH}px`;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, drawW, drawH);

      // Draw the frame
      if (drawSource) {
        ctx.drawImage(drawSource, 0, 0, drawW, drawH);
      }

      // Draw detection boxes
      if (detecting) {
        const data = dataRef.current;
        const hIdx = hoveredRef.current;
        const sIdx = selectedRef.current;
        const detections = data?.detections;
        const newBoxes: typeof boxesRef.current = [];

        if (detections && detections.length > 0) {
          for (let idx = 0; idx < detections.length; idx++) {
            const det = detections[idx];
            const x1 = det.bbox[0] * drawW;
            const y1 = det.bbox[1] * drawH;
            const x2 = det.bbox[2] * drawW;
            const y2 = det.bbox[3] * drawH;
            const color = getColor(det.class_name, det.class_id);
            const isHovered = idx === hIdx;
            const isSelected = idx === sIdx;
            const isHighlighted = isHovered || isSelected;

            newBoxes.push({ x1, y1, x2, y2, det });

            ctx.globalAlpha = isHighlighted ? 1.0 : 0.65 + det.confidence * 0.35;

            if (isHighlighted) {
              ctx.fillStyle = color + '18';
              ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
            }

            // Corner brackets
            const bracketLen = Math.max(8, Math.min(22, (x2 - x1) / 4));
            ctx.strokeStyle = color;
            ctx.lineWidth = isHighlighted ? 3 : 2;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(x1, y1 + bracketLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + bracketLen, y1);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x2 - bracketLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + bracketLen);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x1, y2 - bracketLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + bracketLen, y2);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x2 - bracketLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - bracketLen);
            ctx.stroke();

            // Label
            const label = det.tracker_id !== null
              ? `${det.class_name.toUpperCase()} ${Math.round(det.confidence * 100)}% #${det.tracker_id}`
              : `${det.class_name.toUpperCase()} ${Math.round(det.confidence * 100)}%`;

            ctx.font = '600 11px "Inter", system-ui, sans-serif';
            const labelW = ctx.measureText(label).width + 10;
            const labelH = 18;
            ctx.fillStyle = color;
            ctx.globalAlpha = isHighlighted ? 1.0 : 0.92;

            const labelY = y1 > 30 ? y1 - labelH - 6 : y2 + 6;
            ctx.beginPath();
            if (typeof ctx.roundRect === 'function') {
              ctx.roundRect(x1, labelY, labelW, labelH, 3);
            } else {
              ctx.rect(x1, labelY, labelW, labelH);
            }
            ctx.fill();

            ctx.fillStyle = getContrastColor(color);
            ctx.globalAlpha = 1.0;
            ctx.textBaseline = 'middle';
            ctx.fillText(label, x1 + 5, labelY + labelH / 2);

            // Confidence bar
            if (det.confidence > 0.3) {
              const barY = y2 + 4;
              ctx.fillStyle = '#1a1a1a';
              ctx.globalAlpha = 0.8;
              ctx.fillRect(x1, barY, x2 - x1, 3);
              ctx.fillStyle = color;
              ctx.globalAlpha = 1.0;
              ctx.fillRect(x1, barY, (x2 - x1) * det.confidence, 3);
            }

            ctx.globalAlpha = 1.0;
          }
        }
        boxesRef.current = newBoxes;
      }

    } catch (e) {
      console.error('DetectionView render error:', e);
    }

    rafRef.current = requestAnimationFrame(render);
  };

    rafRef.current = requestAnimationFrame(render);
    return () => { running = false; cancelAnimationFrame(rafRef.current); };
  }, [detecting, frameCanvas, videoElement]);

  // ── Mouse interaction ──────────────────────────────────────────────
  const hitTest = useCallback((e: React.MouseEvent<HTMLCanvasElement>): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    for (let i = boxesRef.current.length - 1; i >= 0; i--) {
      const b = boxesRef.current[i];
      if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) return i;
    }
    return null;
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const idx = hitTest(e);
    setHoveredIdx(idx);
    const canvas = canvasRef.current;
    if (canvas) canvas.style.cursor = idx !== null ? 'pointer' : 'default';
  }, [hitTest]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const idx = hitTest(e);
    setSelectedIdx(prev => prev === idx ? null : idx);
  }, [hitTest]);

  const handleMouseLeave = useCallback(() => { setHoveredIdx(null); }, []);

  return (
    <div className="detection-view">
      <div className="frame-container">
        <canvas
          ref={canvasRef}
          className="frame-canvas"
          onMouseMove={handleMouseMove}
          onClick={handleClick}
          onMouseLeave={handleMouseLeave}
        />
        {/* Loading state: detecting but no data yet */}
        {detecting && !detectionData && (
          <div className="detection-loading">
            <div className="spinner" />
            <p>Waiting for camera frames...</p>
          </div>
        )}
        <div className="frame-overlay">
          {!detecting && <span className="badge badge-source">CAMERA PREVIEW</span>}
          {detecting && detectionData && (
            <>
              <span className="badge badge-source">CAMERA</span>
              <span className="badge badge-fps">{detectionData.fps.toFixed(1)} FPS</span>
              <span className="badge badge-objects">
                {detectionData.active_tracks_count} active · {detectionData.total_unique_seen} unique
              </span>
              {detectionData.inference_ms > 0 && (
                <span className="badge badge-latency">{detectionData.inference_ms.toFixed(0)}ms infer</span>
              )}
              {detectionData.model && (
                <span className="badge badge-model">{detectionData.model.toUpperCase()}</span>
              )}
            </>
          )}
        </div>
        {detectionData && detectionData.detections.length > 0 && (
          <div className="detection-summary">
            <div className="summary-item">
              <span className="summary-icon" />
              <span className="summary-count">{detectionData.detections.length}</span>
              <span className="summary-label">objects detected</span>
            </div>
          </div>
        )}
        {selectedDet && (
          <div className="detection-detail">
            <div className="detail-header">
              <span className="detail-class" style={{ color: getColor(selectedDet.class_name, selectedDet.class_id) }}>
                {selectedDet.class_name}
              </span>
              {selectedDet.tracker_id !== null && (
                <span className="detail-tracker">#{selectedDet.tracker_id}</span>
              )}
            </div>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="detail-label">Confidence</span>
                <span className={`detail-value ${selectedDet.confidence >= 0.7 ? 'conf-high' : selectedDet.confidence >= 0.4 ? 'conf-medium' : 'conf-low'}`}>
                  {Math.round(selectedDet.confidence * 100)}%
                </span>
              </div>
              {selectedDet.tracking_duration_frames != null && selectedDet.tracking_duration_frames > 0 && (
                <div className="detail-item">
                  <span className="detail-label">Tracking</span>
                  <span className="detail-value">{(selectedDet.tracking_duration_frames / 10).toFixed(1)}s</span>
                </div>
              )}
              <div className="detail-item">
                <span className="detail-label">BBox</span>
                <span className="detail-value detail-mono">
                  [{selectedDet.bbox.map(b => b.toFixed(2)).join(', ')}]
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DetectionView;
