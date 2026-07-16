import React, { useState, useEffect, useMemo } from 'react';
import type { DetectionData, DetectorStats } from '../types';

interface Props {
  detectionData: DetectionData | null;
  fetchStats: () => Promise<DetectorStats | null>;
}

const StatsDashboard: React.FC<Props> = ({ detectionData, fetchStats }) => {
  const [stats, setStats] = useState<DetectorStats | null>(null);
  const [topClass, setTopClass] = useState<string>('—');

  // Update top class from session counts
  useEffect(() => {
    if (detectionData?.session_counts) {
      const entries = Object.entries(detectionData.session_counts);
      if (entries.length > 0) {
        entries.sort(([, a], [, b]) => b - a);
        setTopClass(`${entries[0][0]} (${entries[0][1]})`);
      }
    }
  }, [detectionData?.session_counts]);

  // Fetch stats periodically
  useEffect(() => {
    const interval = setInterval(async () => {
      const s = await fetchStats();
      if (s) setStats(s);
    }, 3000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  // Compute top-5 detected classes
  const top5 = useMemo(() => {
    if (!detectionData?.session_counts) return [];
    return Object.entries(detectionData.session_counts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);
  }, [detectionData?.session_counts]);

  const uptimeStr = useMemo(() => {
    if (!stats?.uptime_seconds) return '0s';
    const s = stats.uptime_seconds;
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    if (h > 0) return `${h}h ${m}m ${sec}s`;
    if (m > 0) return `${m}m ${sec}s`;
    return `${sec}s`;
  }, [stats?.uptime_seconds]);

  const totalDetected = useMemo(() => {
    if (detectionData?.session_counts) {
      return Object.values(detectionData.session_counts).reduce((a, b) => a + b, 0);
    }
    return stats?.total_objects_detected ?? 0;
  }, [detectionData?.session_counts, stats?.total_objects_detected]);

  const uniqueClasses = useMemo(() => {
    if (detectionData?.session_counts) {
      return Object.keys(detectionData.session_counts).length;
    }
    return stats?.unique_classes_detected ?? 0;
  }, [detectionData?.session_counts, stats?.unique_classes_detected]);

  return (
    <div className="stats-dashboard">
      <div className="panel-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 3v18h18" />
            <path d="M18.7 8l-5.1 5.2-2.8-2.8L7 14.3" />
          </svg>
          Live Stats
        </h3>
      </div>

      <div className="stats-grid">
        {/* Metric cards */}
        <div className="stat-card highlight">
          <div className="stat-value">{detectionData?.total_unique_seen ?? 0}</div>
          <div className="stat-label">Unique Objects</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{detectionData?.active_tracks_count ?? 0}</div>
          <div className="stat-label">Active Tracks</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{detectionData?.fps.toFixed(1) ?? '0.0'}</div>
          <div className="stat-label">FPS</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{detectionData?.inference_ms.toFixed(0) ?? '—'}<small>ms</small></div>
          <div className="stat-label">Inference</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.frames_processed ?? 0}</div>
          <div className="stat-label">Frames</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{uptimeStr}</div>
          <div className="stat-label">Uptime</div>
        </div>
      </div>

      {/* Top 5 Classes */}
      {top5.length > 0 && (
        <div className="top-classes">
          <h4>Most Detected</h4>
          {top5.map(([name, count], i) => (
            <div key={name} className="top-class-item">
              <span className="top-class-rank">#{i + 1}</span>
              <span className="top-class-name">{name}</span>
              <span className="top-class-count">{count}</span>
            </div>
          ))}
        </div>
      )}

      {/* Device Info */}
      {stats && (
        <div className="device-info">
          <div className="device-item">
            <span className="device-label">Device</span>
            <span className="device-value code">{stats.device}</span>
          </div>
          <div className="device-item">
            <span className="device-label">Confidence</span>
            <span className="device-value">{stats.conf_threshold.toFixed(2)}</span>
          </div>
          <div className="device-item">
            <span className="device-label">Model</span>
            <span className="device-value">YOLOv8n (Traffic)</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default StatsDashboard;
