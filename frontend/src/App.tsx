import React, { useCallback, useEffect, useState } from 'react';
import { useDetector } from './hooks/useDetector';
import DetectionView from './components/DetectionView';
import ObjectCounter from './components/ObjectCounter';
import ControlPanel from './components/ControlPanel';
import StatsDashboard from './components/StatsDashboard';
import CongestionBadge from './components/CongestionBadge';
import CategoryBreakdown from './components/CategoryBreakdown';
import TrendChart from './components/TrendChart';
import TrackerInfoPanel from './components/TrackerInfoPanel';
import type { ModelsResponse } from './types';
import './App.css';

const App: React.FC = () => {
  const {
    cameraState, cameraError, requestCamera, videoElement,
    startDetection, stopDetection, detecting, detectionData, frameCanvas,
    connected, fetchStats, fetchModels, switchModel,
  } = useDetector();

  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);

  useEffect(() => {
    if (connected) fetchModels().then(setModelsData);
  }, [connected, fetchModels]);

  const handleConfig = useCallback(
    (conf: { conf?: number; iou?: number; model?: string; class_filter?: string; imgsz?: number }) => {
      fetch('/api/config', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(conf),
      }).catch(() => {});
    }, []
  );

  const handleReset = useCallback(() => {
    fetch('/api/reset', { method: 'POST' }).catch(() => {});
  }, []);

  const handleExportCSV = useCallback(() => {
    window.open('/api/export/csv', '_blank');
  }, []);

  // ── Loading ──────────────────────────────────────────────────────────
  if (!connected) {
    return (
      <div className="app">
        <div className="permission-gate">
          <div className="permission-card">
            <div className="permission-icon"><div className="spinner" /></div>
            <h2>Connecting to Server...</h2>
            <p className="permission-desc">Starting YOLOv8 detection engine on GPU</p>
          </div>
        </div>
      </div>
    );
  }

  // ── Camera error ─────────────────────────────────────────────────────
  if (cameraState === 'error') {
    return (
      <div className="app">
        <div className="permission-gate">
          <div className="permission-card">
            <div className="permission-icon">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            </div>
            <h2>Camera Unavailable</h2>
            <p className="permission-desc">{cameraError}</p>
            <div className="permission-actions">
              <button className="btn btn-primary" onClick={requestCamera}>Try Again</button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Camera permission gate ───────────────────────────────────────────
  if (cameraState === 'idle' || cameraState === 'requesting') {
    return (
      <div className="app">
        <div className="permission-gate">
          <div className="permission-card">
            <div className="permission-icon">
              {cameraState === 'requesting' ? (
                <div className="spinner" />
              ) : (
                <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                  <circle cx="12" cy="13" r="4" />
                </svg>
              )}
            </div>
            <h2>Camera Access Required</h2>
            <p className="permission-desc">
              Object Detection uses your camera to detect objects in real-time.
              <br />Your camera feed is processed locally and never leaves your device.
            </p>
            {cameraState === 'requesting' ? (
              <div className="permission-loading">
                <p>Waiting for camera permission...</p>
                <p className="hint">Check your browser's permission prompt</p>
              </div>
            ) : (
              <div className="permission-actions">
                <button className="btn btn-primary btn-lg" onClick={requestCamera}>
                  Grant Camera Access
                </button>
              </div>
            )}
            <div className="permission-footer">
              <div className="privacy-badge">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
                <span>100% local processing — nothing leaves your device</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Main detection UI ───────────────────────────────────────────────
  const modelDisplayName = modelsData?.current?.toUpperCase() || 'YOLOv8';

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" />
            </svg>
            <h1>Object Detection</h1>
          </div>
          <span className="header-subtitle">Real-Time Camera Detection</span>
        </div>
        <div className="header-right">
          <div className={`connection-badge ${connected ? 'online' : 'offline'}`}>
            <span className="badge-dot" />
            {connected ? 'Connected' : 'Connecting...'}
          </div>
          <div className="model-badge">{modelDisplayName}</div>
        </div>
      </header>

      <div className="app-layout">
        <aside className="sidebar sidebar-left">
          <ControlPanel
            connected={connected}
            detecting={detecting}
            onStart={startDetection}
            onStop={stopDetection}
            onConfig={handleConfig}
            onReset={handleReset}
            onExport={handleExportCSV}
            currentConf={0.35}
            modelsData={modelsData}
            onModelSwitch={switchModel}
          />
          <CongestionBadge detectionData={detectionData} />
          <CategoryBreakdown detectionData={detectionData} />
        </aside>

        <main className="main-content">
          <DetectionView
            videoElement={videoElement}
            frameCanvas={frameCanvas}
            detecting={detecting}
            detectionData={detectionData}
          />
          <TrendChart detectionData={detectionData} />
        </main>

        <aside className="sidebar sidebar-right">
          <StatsDashboard detectionData={detectionData} fetchStats={fetchStats} />
          <TrackerInfoPanel fetchStats={fetchStats} />
          <ObjectCounter detectionData={detectionData} />
        </aside>
      </div>

      <footer className="app-footer">
        <span>Powered by {modelDisplayName} | 80 COCO Classes</span>
        <span className="footer-sep">|</span>
        <span>Camera capture at {detectionData?.fps.toFixed(1) ?? '—'} FPS</span>
      </footer>
    </div>
  );
};

export default App;
