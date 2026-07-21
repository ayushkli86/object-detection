import React, { useEffect, useRef, useCallback } from 'react';
import type { CameraSource } from '../types';

interface Props {
  cameras: CameraSource[];
  activeCameraId: string;
  onSelect: (id: string) => void;
  onClose: () => void;
}

const CameraGrid: React.FC<Props> = ({ cameras, activeCameraId, onSelect, onClose }) => {
  const canvasRefs = useRef<Map<string, HTMLCanvasElement>>(new Map());
  const imgRefs = useRef<Map<string, HTMLImageElement>>(new Map());
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevUrls = useRef<Map<string, string>>(new Map());

  // Poll all camera frames
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      if (cancelled) return;
      try {
        const res = await fetch('/api/all-frames');
        if (!res.ok) return;
        const data = await res.json();

        for (const [camId, camData] of Object.entries(data.cameras) as [string, any][]) {
          if (cancelled) return;
          const canvas = canvasRefs.current.get(camId);
          if (!canvas) continue;

          // Draw detection info on canvas
          const ctx = canvas.getContext('2d');
          if (!ctx) continue;

          // Dark background
          ctx.fillStyle = '#334155';
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          // Camera name
          ctx.fillStyle = '#f1f5f9';
          ctx.font = 'bold 14px Inter, system-ui, sans-serif';
          ctx.fillText(camData.name || camId, 10, 25);

          // Status
          const count = camData.total_objects || 0;
          const fps = camData.fps || 0;
          ctx.font = '12px Inter, system-ui, sans-serif';
          ctx.fillStyle = '#4ade80';
          ctx.fillText(`${count} objects  |  ${fps} FPS`, 10, 48);

          // Draw detection boxes
          const detections = camData.detections || [];
          const getDetColor = (name: string) => {
            let h = 0;
            for (let i = 0; i < name.length; i++) h = ((h << 5) - h + name.charCodeAt(i)) | 0;
            return `hsl(${Math.abs(h) % 360}, 70%, 55%)`;
          };
          const fallbackColors = ['#3b82f6', '#4ade80', '#f59e0b', '#8b5cf6'];

          for (const det of detections) {
            const [x1, y1, x2, y2] = det.bbox;
            const color = getDetColor(det.class_name);
            const bx = x1 * canvas.width;
            const by = y1 * canvas.height;
            const bw = (x2 - x1) * canvas.width;
            const bh = (y2 - y1) * canvas.height;

            // Box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(bx, by, bw, bh);

            // Label
            ctx.fillStyle = color;
            ctx.font = '10px Inter, system-ui, sans-serif';
            const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;
            ctx.fillText(label, bx + 4, by - 4 > 10 ? by - 4 : by + 12);
          }

          if (detections.length === 0) {
            ctx.fillStyle = '#64748b';
            ctx.font = '13px Inter, system-ui, sans-serif';
            ctx.fillText('No objects detected', canvas.width / 2 - 60, canvas.height / 2);
          }

          // Active indicator
          if (camId === data.active_camera) {
            ctx.strokeStyle = '#4ade80';
            ctx.lineWidth = 3;
            ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
            ctx.fillStyle = '#4ade80';
            ctx.font = 'bold 11px Inter, system-ui, sans-serif';
            ctx.fillText('ACTIVE', canvas.width - 55, 20);
          }
        }
      } catch (e) {
        console.warn('Camera grid poll failed:', e);
      }
    };

    poll();
    pollRef.current = setInterval(poll, 200);
    return () => { cancelled = true; if (pollRef.current) clearInterval(pollRef.current); };
  }, [cameras]);

  const handleSelect = useCallback((id: string) => {
    onSelect(id);
    onClose();
  }, [onSelect, onClose]);

  return (
    <div className="camera-grid-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="camera-grid-modal">
        <div className="camera-grid-header">
          <h2>Connected Cameras</h2>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>X Close</button>
        </div>
        <div className="camera-grid-tiles">
          {/* Local camera tile */}
          <div
            className={`camera-tile ${activeCameraId === 'local' ? 'active' : ''}`}
            onClick={() => handleSelect('local')}
          >
            <canvas
              ref={(el) => { if (el) canvasRefs.current.set('local', el); }}
              width={320}
              height={240}
              className="camera-tile-canvas"
            />
            <div className="camera-tile-info">
              <span>Local Camera</span>
              {activeCameraId === 'local' && <span className="active-badge">ACTIVE</span>}
            </div>
          </div>
          {/* Remote camera tiles */}
          {cameras.filter(c => c.type === 'remote').map(cam => (
            <div
              key={cam.id}
              className={`camera-tile ${cam.id === activeCameraId ? 'active' : ''}`}
              onClick={() => handleSelect(cam.id)}
            >
              <canvas
                ref={(el) => { if (el) canvasRefs.current.set(cam.id, el); }}
                width={320}
                height={240}
                className="camera-tile-canvas"
              />
              <div className="camera-tile-info">
                <span>{cam.name}</span>
                {cam.id === activeCameraId && <span className="active-badge">ACTIVE</span>}
              </div>
            </div>
          ))}
          {cameras.filter(c => c.type === 'remote').length === 0 && (
            <div className="camera-grid-empty">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5">
                <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/>
                <line x1="12" y1="18" x2="12.01" y2="18"/>
              </svg>
              <p>No phones connected yet</p>
              <p className="hint">Open <code>http://<span style={{color:'var(--accent-cyan)'}}>{window.location.hostname}:8765/mobile.html</span></code> on your phone</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CameraGrid;
