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
import type { ModelsResponse, CameraSource } from './types';
import CameraGrid from './components/CameraGrid';
import './App.css';

const App: React.FC = () => {
  const {
    cameraState, cameraError, requestCamera, videoElement,
    startDetection, stopDetection, detecting, detectionData, frameCanvas, remoteFrameUrl, lanIp,
    connected, fetchStats, fetchModels, switchModel,
    activeCameraId, setActiveCamera, availableCameras, fetchCameras,
    updateFps, currentFps,
  } = useDetector();

  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [initialConf, setInitialConf] = useState(0.35);
  const [showGrid, setShowGrid] = useState(false);

  useEffect(() => {
    if (connected) {
      fetchModels().then(setModelsData);
      fetch('/api/stats').then(r => r.json()).then(s => {
        if (s?.conf_threshold) setInitialConf(s.conf_threshold);
      }).catch(() => {});
    }
  }, [connected, fetchModels]);

  // Poll camera list for badge count
  useEffect(() => {
    if (!connected) return;
    const poll = async () => { await fetchCameras(); };
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [connected, fetchCameras]);

  const handleConfig = useCallback(
    (conf: { conf?: number; iou?: number; model?: string; class_filter?: string; imgsz?: number }) => {
      fetch('/api/config', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(conf),
      }).catch(e => console.warn('Config update failed:', e));
    }, []
  );

  const handleReset = useCallback(() => {
    fetch('/api/reset', { method: 'POST' }).catch(e => console.warn('Reset failed:', e));
  }, []);

  const handleExportCSV = useCallback(() => {
    window.open('/api/export/csv', '_blank');
  }, []);

  const handleSwitchCamera = async (id: string) => {
    await setActiveCamera(id);
    fetchCameras();
  };

  const remoteCount = availableCameras.filter(c => c.type === 'remote').length;

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
              Traffic Object Detection System uses your camera to detect vehicles and traffic objects in real-time.
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
                <div className="phone-connect">
                  <div className="phone-connect-header">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/>
                      <line x1="12" y1="18" x2="12.01" y2="18"/>
                    </svg>
                    <span>Stream from your phone</span>
                  </div>
                  <p className="phone-connect-desc">
                    Open <code>http://{lanIp}/mobile.html</code> on your phone's browser
                  </p>
                </div>
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
  const activeCamName = availableCameras.find(c => c.id === activeCameraId)?.name || 'Local Camera';
  const isRemote = activeCameraId !== 'local';

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" />
            </svg>
            <h1>Traffic Object Detection System</h1>
          </div>
          <span className="header-subtitle">Real-Time Vehicle Detection and Traffic Analysis for Nepal</span>
        </div>
        <div className="header-right">
          {/* Cameras button → opens grid modal */}
          <button className="cam-selector-btn" onClick={() => setShowGrid(true)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" />
            </svg>
            <span>Cameras</span>
            {remoteCount > 0 && <span className="cam-badge">{remoteCount}</span>}
          </button>
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
            currentConf={initialConf}
            modelsData={modelsData}
            onModelSwitch={switchModel}
            currentFps={currentFps}
            onFpsChange={updateFps}
          />
          <CongestionBadge detectionData={detectionData} />
          <CategoryBreakdown detectionData={detectionData} />
          <div className="phone-connect-card">
            <div className="panel-header">
              <h3>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/>
                  <line x1="12" y1="18" x2="12.01" y2="18"/>
                </svg>
                Phone Stream
              </h3>
            </div>
            <div className="phone-connect-body">
              <p>Open on your phone:</p>
              <code className="phone-url">http://{lanIp}/mobile.html</code>
            </div>
          </div>
        </aside>

        <main className="main-content">
          <DetectionView
            videoElement={activeCameraId === 'local' ? videoElement : null}
            frameCanvas={activeCameraId === 'local' ? frameCanvas : null}
            detecting={detecting}
            detectionData={detectionData}
            remoteFrameUrl={activeCameraId !== 'local' ? remoteFrameUrl : null}
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
        <span>Powered by {modelDisplayName} | Traffic Classes</span>
        <span className="footer-sep">|</span>
        <span>{isRemote ? '[Phone]' : '[Cam]'} {activeCamName} at {detectionData?.fps.toFixed(1) ?? '--'} FPS</span>
      </footer>

      {/* Camera Grid Modal */}
      {showGrid && (
        <CameraGrid
          cameras={availableCameras}
          activeCameraId={activeCameraId}
          onSelect={handleSwitchCamera}
          onClose={() => setShowGrid(false)}
        />
      )}
    </div>
  );
};

export default App;
