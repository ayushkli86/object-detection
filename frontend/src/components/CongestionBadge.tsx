import React, { useMemo } from 'react';
import type { DetectionData, CongestionLevel } from '../types';

interface Props {
  detectionData: DetectionData | null;
}

const LEVELS: { level: CongestionLevel; label: string; color: string; bg: string; icon: string; max: number }[] = [
  { level: 'low',      label: 'Low',      color: '#22c55e', bg: 'rgba(34,197,94,0.12)',  icon: '●', max: 15 },
  { level: 'medium',   label: 'Medium',   color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', icon: '●●', max: 35 },
  { level: 'high',     label: 'High',     color: '#f97316', bg: 'rgba(249,115,22,0.12)', icon: '●●●', max: 60 },
  { level: 'severe',   label: 'Severe',   color: '#ef4444', bg: 'rgba(239,68,68,0.12)',  icon: '●●●●', max: Infinity },
];

function getClassify(count: number): typeof LEVELS[number] {
  for (const l of LEVELS) {
    if (count <= l.max) return l;
  }
  return LEVELS[LEVELS.length - 1];
}

const CongestionBadge: React.FC<Props> = ({ detectionData }) => {
  const activeCount = detectionData?.active_tracks_count ?? 0;
  const totalSeen = detectionData?.total_unique_seen ?? 0;

  const level = useMemo(() => getClassify(activeCount), [activeCount]);
  const prevLevel = useMemo(() => getClassify(Math.max(0, activeCount - 5)), [activeCount]);

  const isEscalating = LEVELS.indexOf(level) > LEVELS.indexOf(prevLevel);

  return (
    <div className="congestion-badge" style={{ borderColor: level.color }}>
      <div className="congestion-header">
        <span className="congestion-icon" style={{ color: level.color }}>{level.icon}</span>
        <span className="congestion-label" style={{ color: level.color }}>{level.label}</span>
      </div>
      <div className="congestion-count">
        <span className="congestion-number">{activeCount}</span>
        <span className="congestion-sub">active objects</span>
      </div>
      <div className="congestion-bar-track">
        <div
          className="congestion-bar-fill"
          style={{
            width: `${Math.min(100, (activeCount / 60) * 100)}%`,
            backgroundColor: level.color,
          }}
        />
      </div>
      {isEscalating && (
        <div className="congestion-alert">
          <span className="congestion-alert-dot" style={{ backgroundColor: level.color }} />
          Traffic increasing
        </div>
      )}
    </div>
  );
};

export default CongestionBadge;
