import React, { useState, useEffect } from 'react';

interface Props {
  connected: boolean;
  detecting: boolean;
  onStart: () => void;
  onStop: () => void;
  onConfig: (conf: { conf?: number; iou?: number }) => void;
  onReset: () => void;
  currentConf: number;
  currentFps: number;
}

const ControlPanel: React.FC<Props> = ({
  connected,
  detecting,
  onStart,
  onStop,
  onConfig,
  onReset,
  currentConf,
  currentFps,
}) => {
  const [confThresh, setConfThresh] = useState(currentConf);
  const [targetFps, setTargetFps] = useState(currentFps);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    setConfThresh(currentConf);
  }, [currentConf]);

  useEffect(() => {
    setTargetFps(currentFps);
  }, [currentFps]);

  const handleStartStop = () => {
    if (detecting) onStop();
    else onStart();
  };

  const handleConfChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setConfThresh(val);
    onConfig({ conf: val });
  };

  return (
    <div className="control-panel">
      <div className="panel-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
          Controls
        </h3>
      </div>

      <div className="controls-body">
        {/* Start/Stop Button */}
        <button
          className={`btn btn-${detecting ? 'danger' : 'primary'} btn-lg`}
          onClick={handleStartStop}
          disabled={!connected}
        >
          {detecting ? (
            <>
              <span className="btn-icon pulse-dot" />
              Stop Detection
            </>
          ) : (
            <>
              <span className="btn-icon play-icon">&#9654;</span>
              Start Detection
            </>
          )}
        </button>

        {/* Connection status */}
        <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          <span className="status-text">{connected ? 'Connected' : 'Disconnected'}</span>
        </div>

        {/* Confidence Threshold */}
        <div className="control-group">
          <label className="control-label">
            Confidence
            <span className="control-value">{confThresh.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={confThresh}
            onChange={handleConfChange}
            className="slider"
          />
          <div className="slider-labels">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>

        {/* Advanced toggle */}
        <button
          className="btn btn-ghost btn-sm"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced
        </button>

        {showAdvanced && (
          <div className="advanced-controls">
            {/* Target FPS */}
            <div className="control-group">
              <label className="control-label">
                Target FPS
                <span className="control-value">{targetFps}</span>
              </label>
              <input
                type="range"
                min="1"
                max="30"
                step="1"
                value={targetFps}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  setTargetFps(val);
                  fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_fps: val }),
                  }).catch(() => {});
                }}
                className="slider"
              />
            </div>

            {/* Reset Counts */}
            <button className="btn btn-warning btn-sm" onClick={onReset}>
              Reset Object Counts
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlPanel;
