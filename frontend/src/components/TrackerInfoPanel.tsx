import React, { useState, useEffect, useRef } from 'react';
import type { DetectorStats } from '../types';

interface Props {
  fetchStats: () => Promise<DetectorStats | null>;
}

const TRACKER_INFO: Record<string, { name: string; desc: string; color: string }> = {
  botsort: { name: 'BoT-SORT', desc: 'Appearance + Motion (ReID)', color: '#4ecdc4' },
  bytetrack: { name: 'ByteTrack', desc: 'IoU-based matching', color: '#5b8af5' },
};

const TrackerInfoPanel: React.FC<Props> = ({ fetchStats }) => {
  const [stats, setStats] = useState<DetectorStats | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const s = await fetchStats();
      if (!cancelled && s) setStats(s);
      if (!cancelled) {
        timerRef.current = setTimeout(load, 3000);
      }
    };
    load();
    return () => { cancelled = true; if (timerRef.current) clearTimeout(timerRef.current); };
  }, [fetchStats]);

  if (!stats) return null;

  const tracker = TRACKER_INFO[stats.tracker] || { name: stats.tracker, desc: 'Custom', color: '#64748b' };
  const stabilityPct = Math.round(stats.tracking_stability * 100);
  const stabilityColor =
    stabilityPct >= 90 ? '#6b7c5e' : stabilityPct >= 70 ? '#b8860b' : '#8b3a3a';

  const modelName = stats.model || 'yolov8l';

  return (
    <div className="tracker-info">
      <div className="panel-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
          Detection
        </h3>
        <span
          className="tracker-type-badge"
          style={{ color: tracker.color, borderColor: tracker.color }}
        >
          {tracker.name}
        </span>
      </div>

      <div className="tracker-body">
        <div className="tracker-desc">{tracker.desc}</div>

        {/* Model info */}
        <div className="tracker-model-info">
          <span className="tracker-model-name">{modelName.toUpperCase()}</span>
          <span className="tracker-model-detail">
            {stats.model_info?.params || '—'} params | mAP {stats.model_info?.map || '—'}
          </span>
          <span className="tracker-model-detail">
            Input: {stats.imgsz || 640}px | {stats.active_classes || 80} classes
          </span>
          {stats.model_info?.description && (
            <span className="tracker-model-detail" style={{color: '#22c55e'}}>
              {stats.model_info.description}
            </span>
          )}
        </div>

        {/* Stability meter */}
        <div className="tracker-meter">
          <div className="tracker-meter-header">
            <span className="tracker-meter-label">Stability</span>
            <span className="tracker-meter-value" style={{ color: stabilityColor }}>
              {stabilityPct}%
            </span>
          </div>
          <div className="tracker-meter-track">
            <div
              className="tracker-meter-fill"
              style={{ width: `${stabilityPct}%`, backgroundColor: stabilityColor }}
            />
          </div>
        </div>

        {/* Stats grid */}
        <div className="tracker-stats-grid">
          <div className="tracker-stat">
            <span className="tracker-stat-value">{stats.total_tracks_created}</span>
            <span className="tracker-stat-label">Tracks Created</span>
          </div>
          <div className="tracker-stat">
            <span className="tracker-stat-value">{stats.total_id_switches}</span>
            <span className="tracker-stat-label">ID Switches</span>
          </div>
          <div className="tracker-stat">
            <span className="tracker-stat-value">{stats.conf_threshold.toFixed(2)}</span>
            <span className="tracker-stat-label">Confidence</span>
          </div>
          <div className="tracker-stat">
            <span className="tracker-stat-value">{stats.iou_threshold.toFixed(2)}</span>
            <span className="tracker-stat-label">IoU Threshold</span>
          </div>
        </div>

        {/* Device */}
        <div className="tracker-device">
          <span className="tracker-device-label">Device</span>
          <span className="tracker-device-value">{stats.device}</span>
        </div>
      </div>
    </div>
  );
};

export default TrackerInfoPanel;
