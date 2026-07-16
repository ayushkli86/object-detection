import React, { useEffect, useRef, useState } from 'react';
import type { DetectionData } from '../types';

interface Props {
  detectionData: DetectionData | null;
}

const MAX_POINTS = 60;

const TrendChart: React.FC<Props> = ({ detectionData }) => {
  const [history, setHistory] = useState<number[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  // Append current active_tracks_count to history
  useEffect(() => {
    const val = detectionData?.active_tracks_count ?? 0;
    setHistory((prev) => {
      const next = [...prev, val];
      return next.length > MAX_POINTS ? next.slice(next.length - MAX_POINTS) : next;
    });
  }, [detectionData?.active_tracks_count]);

  const maxVal = Math.max(1, ...history);
  const avgVal = history.length > 0
    ? Math.round(history.reduce((a, b) => a + b, 0) / history.length)
    : 0;
  const currentVal = history.length > 0 ? history[history.length - 1] : 0;

  // Generate SVG path
  const width = 260;
  const height = 80;
  const padding = 4;

  const points = history.map((val, i) => {
    const x = padding + (i / Math.max(1, history.length - 1)) * (width - padding * 2);
    const y = height - padding - (val / maxVal) * (height - padding * 2);
    return `${x},${y}`;
  });

  const linePath = points.length > 1 ? `M ${points.join(' L ')}` : '';
  const areaPath = points.length > 1
    ? `M ${padding},${height - padding} L ${points.join(' L ')} L ${padding + ((history.length - 1) / Math.max(1, history.length - 1)) * (width - padding * 2)},${height - padding} Z`
    : '';

  return (
    <div className="trend-chart">
      <div className="panel-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          Detection Trend
        </h3>
        <span className="trend-badge">{history.length} samples</span>
      </div>

      <div className="trend-stats">
        <div className="trend-stat">
          <span className="trend-stat-value">{currentVal}</span>
          <span className="trend-stat-label">Current</span>
        </div>
        <div className="trend-stat">
          <span className="trend-stat-value">{avgVal}</span>
          <span className="trend-stat-label">Average</span>
        </div>
        <div className="trend-stat">
          <span className="trend-stat-value">{maxVal}</span>
          <span className="trend-stat-label">Peak</span>
        </div>
      </div>

      <div className="trend-svg-container" ref={containerRef}>
        {history.length > 1 ? (
          <svg viewBox={`0 0 ${width} ${height}`} className="trend-svg" preserveAspectRatio="none">
            {/* Grid lines */}
            <line x1={padding} y1={height / 4} x2={width - padding} y2={height / 4} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
            <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
            <line x1={padding} y1={(height * 3) / 4} x2={width - padding} y2={(height * 3) / 4} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />

            {/* Area fill */}
            <path d={areaPath} fill="url(#trendGradient)" opacity="0.3" />

            {/* Line */}
            <path d={linePath} fill="none" stroke="var(--accent-cyan)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

            {/* Current point */}
            {points.length > 0 && (
              <circle
                cx={parseFloat(points[points.length - 1].split(',')[0])}
                cy={parseFloat(points[points.length - 1].split(',')[1])}
                r="3"
                fill="var(--accent-cyan)"
                stroke="var(--bg-primary)"
                strokeWidth="1.5"
              />
            )}

            {/* Gradient definition */}
            <defs>
              <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent-cyan)" stopOpacity="0.4" />
                <stop offset="100%" stopColor="var(--accent-cyan)" stopOpacity="0" />
              </linearGradient>
            </defs>
          </svg>
        ) : (
          <div className="trend-empty">
            <span>Collecting data...</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrendChart;
