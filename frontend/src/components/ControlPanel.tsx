import React, { useState, useEffect, useRef } from 'react';
import type { ModelsResponse } from '../types';

interface Props {
  connected: boolean;
  detecting: boolean;
  onStart: () => void;
  onStop: () => void;
  onConfig: (conf: { conf?: number; iou?: number; model?: string; class_filter?: string; imgsz?: number }) => void;
  onReset: () => void;
  onExport: () => void;
  currentConf: number;
  modelsData: ModelsResponse | null;
  onModelSwitch: (model: string) => Promise<boolean>;
}

const CLASS_FILTERS = [
  { value: 'all', label: 'All Classes' },
  { value: 'vehicles', label: 'Vehicles Only' },
  { value: 'people_animals', label: 'People & Animals' },
  { value: 'objects', label: 'Objects Only' },
];

const ControlPanel: React.FC<Props> = ({
  connected, detecting, onStart, onStop, onConfig, onReset, onExport,
  currentConf, modelsData, onModelSwitch,
}) => {
  const [confThresh, setConfThresh] = useState(currentConf);
  const [selectedModel, setSelectedModel] = useState(modelsData?.current || 'yolov8l');
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [switching, setSwitching] = useState(false);
  const switchingRef = useRef(false);

  useEffect(() => { setConfThresh(currentConf); }, [currentConf]);
  useEffect(() => {
    if (modelsData?.current) setSelectedModel(modelsData.current);
  }, [modelsData?.current]);

  const handleStartStop = () => { detecting ? onStop() : onStart(); };

  const handleConfChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setConfThresh(val);
    onConfig({ conf: val });
  };

  const handleModelChange = async (modelName: string) => {
    if (switchingRef.current) return; // prevent double-click race
    switchingRef.current = true;
    setSwitching(true);
    setSelectedModel(modelName);
    const ok = await onModelSwitch(modelName);
    if (!ok) setSelectedModel(modelsData?.current || 'yolov8l');
    setSwitching(false);
    switchingRef.current = false;
  };

  const handleFilterChange = (filter: string) => {
    setSelectedFilter(filter);
    onConfig({ class_filter: filter });
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
            <><span className="btn-icon pulse-dot" />Stop Detection</>
          ) : (
            <><span className="btn-icon play-icon">&#9654;</span>Start Detection</>
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
            type="range" min="0.05" max="0.95" step="0.05"
            value={confThresh} onChange={handleConfChange} className="slider"
          />
          <div className="slider-labels"><span>Low (more detections)</span><span>High (fewer, precise)</span></div>
        </div>

        {/* Model selector */}
        <div className="control-group">
          <label className="control-label">
            Model
            <span className="control-value">{selectedModel}</span>
          </label>
          <div className="model-selector">
            {modelsData && Object.entries(modelsData.available).map(([name, info]: [string, any]) => (
              <button
                key={name}
                className={`model-btn ${selectedModel === name ? 'active' : ''} ${switching ? 'switching' : ''} ${info.finetuned ? 'finetuned' : ''}`}
                onClick={() => handleModelChange(name)}
                disabled={switching}
                title={`${info.params} params | mAP ${info.map} | ~${info.speed_ms}ms${info.description ? ' | ' + info.description : ''}`}
              >
                <span className="model-btn-name">{name.replace('yolov8', 'v8')}</span>
                <span className="model-btn-info">
                  {info.finetuned && <span className="finetuned-badge">FINETUNED</span>}
                  mAP {info.map}
                </span>
                {info.description && <span className="model-btn-desc">{info.description}</span>}
              </button>
            ))}
          </div>
        </div>

        {/* Class filter */}
        <div className="control-group">
          <label className="control-label">Class Filter</label>
          <div className="filter-chips">
            {CLASS_FILTERS.map(f => (
              <button
                key={f.value}
                className={`filter-chip ${selectedFilter === f.value ? 'active' : ''}`}
                onClick={() => handleFilterChange(f.value)}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced toggle */}
        <button className="btn btn-ghost btn-sm" onClick={() => setShowAdvanced(!showAdvanced)}>
          {showAdvanced ? 'Hide' : 'Show'} Advanced
        </button>

        {showAdvanced && (
          <div className="advanced-controls">
            <div className="control-group">
              <label className="control-label">
                Target FPS
                <span className="control-value">15</span>
              </label>
              <input
                type="range" min="1" max="30" step="1" defaultValue="15"
                className="slider"
                onChange={(e) => {
                  fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_fps: parseInt(e.target.value, 10) }),
                  }).catch(() => {});
                }}
              />
            </div>

            <button className="btn btn-warning btn-sm" onClick={onReset}>
              Reset Object Counts
            </button>

            <button className="btn btn-ghost btn-sm" onClick={onExport}>
              Export History (CSV)
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlPanel;
