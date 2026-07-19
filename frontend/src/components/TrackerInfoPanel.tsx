import React, { useState, useEffect } from 'react';
import type { DetectorStats } from '../types';

interface Props {
  fetchStats: () => Promise<DetectorStats | null>;
}

const TRACKER_INFO: Record<string, { name: string; desc: string; color: string }> = {
  botsort: { name: 'BoT-SORT', desc: 'Appearance + Motion (ReID)', color: '#22d3ee' },
  bytetrack: { name: 'ByteTrack', desc: 'IoU-based matching', color: '#3b82f6' },
};

const TrackerInfoPanel: React.FC<Props> = ({ fetchStats }) => {
  const [stats, setStats] = useState<DetectorStats | null>(null);

  useEffect(() => {
    const load = async () => {
      const s = await fetchStats();
      if (s) setStats(s);
    };
    load();
    const interval = setInterval(load, 3000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  if (!stats) return null;

  const tracker = TRACKER_INFO[stats.tracker] || { name: stats.tracker, desc: 'Custom', color: '#64748b' };
  const stabilityPct = Math.round(stats.tracking_stability * 100);
  const stabilityColor =
    stabilityPct >= 90 ? '#22c55e' : stabilityPct >= 70 ? '#f59e0b' : '#ef4444';

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
