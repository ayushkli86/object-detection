import React, { useMemo } from 'react';
import type { DetectionData } from '../types';

interface Props {
  detectionData: DetectionData | null;
}

/**
 * Dynamic object counter — works with ANY model (COCO 80, Open Images 601, etc.)
 * Colors are generated deterministically from class names, not hardcoded.
 */

// Known category colors for common objects (used as hints, not restrictions)
const KNOWN_COLORS: Record<string, string> = {
  person: '#3BB8EB', car: '#56CF63', truck: '#56CF63', bus: '#56CF63',
  motorcycle: '#F79646', bicycle: '#F79646',
  cat: '#C850C0', dog: '#C850C0', bird: '#C850C0', horse: '#C850C0',
  cow: '#C850C0', sheep: '#C850C0',
  chair: '#82A2E6', couch: '#82A2E6', bed: '#82A2E6', table: '#82A2E6',
  bottle: '#FF8282', cup: '#FF8282', bowl: '#FF8282',
  laptop: '#FFC882', 'cell phone': '#FFC882', tv: '#FFC882', keyboard: '#FFC882',
  book: '#FFFF82', clock: '#FFFF82', vase: '#FFFF82',
  backpack: '#F79646', 'handbag': '#F79646', umbrella: '#4ecdc4',
  'traffic light': '#e85454', 'stop sign': '#e85454',
};

function getColor(name: string): string {
  if (KNOWN_COLORS[name]) return KNOWN_COLORS[name];
  // Deterministic color from name hash — works for ANY class name
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
  }
  return `hsl(${Math.abs(hash) % 360}, 65%, 55%)`;
}

const ObjectCounter: React.FC<Props> = ({ detectionData }) => {
  // Session counts: unique objects per class (each tracker_id counted once)
  const sortedCounts = useMemo(() => {
    if (!detectionData?.session_counts) return [];
    return Object.entries(detectionData.session_counts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 40);
  }, [detectionData?.session_counts]);

  // Currently visible tracker IDs grouped by class
  const activeByClass = useMemo(() => {
    if (!detectionData?.detections) return new Map<string, number[]>();
    const map = new Map<string, number[]>();
    for (const det of detectionData.detections) {
      if (det.tracker_id == null) continue;
      const existing = map.get(det.class_name) || [];
      existing.push(det.tracker_id);
      map.set(det.class_name, existing);
    }
    return map;
  }, [detectionData?.detections]);

  const totalUnique = detectionData?.total_unique_seen ?? 0;
  const activeCount = detectionData?.active_tracks_count ?? 0;
  const isEmpty = sortedCounts.length === 0;

  return (
    <div className="object-counter">
      <div className="panel-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
          Object Counter
        </h3>
        {detectionData && (
          <span className="total-badge">{totalUnique} unique</span>
        )}
      </div>

      {/* Active tracks summary */}
      {detectionData && activeCount > 0 && (
        <div className="tracking-summary">
          <span className="tracking-label">{activeCount} visible now</span>
        </div>
      )}

      <div className="counter-list">
        {isEmpty ? (
          <div className="counter-empty">
            <p>No objects detected yet</p>
            <p className="hint">Start detection to begin counting</p>
          </div>
        ) : (
          sortedCounts.map(([name, count]) => {
            const maxCount = sortedCounts[0][1];
            const barWidth = (count / maxCount) * 100;
            const color = getColor(name);
            const activeIds = activeByClass.get(name) || [];
            const isActive = activeIds.length > 0;

            return (
              <div key={name} className={`counter-item ${isActive ? 'active' : ''}`}>
                <div className="counter-info">
                  <span
                    className="counter-dot"
                    style={{ backgroundColor: color }}
                  />
                  <span className="counter-name">{name}</span>
                  {isActive && (
                    <span className="counter-live-pulse" title={`Active: #${activeIds.join(', #')}`} />
                  )}
                </div>
                <div className="counter-bar-container">
                  <div
                    className="counter-bar"
                    style={{
                      width: `${barWidth}%`,
                      backgroundColor: color,
                    }}
                  />
                </div>
                <span className="counter-count">{count}</span>
                {activeIds.length > 0 && (
                  <span className="counter-tracker-ids">
                    {activeIds.map(id => `#${id}`).join(' ')}
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default ObjectCounter;
