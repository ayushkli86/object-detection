import React, { useMemo } from 'react';
import type { DetectionData } from '../types';

interface Props {
  detectionData: DetectionData | null;
}

interface CategoryGroup {
  name: string;
  label: string;
  color: string;
  classes: string[];
}

const CATEGORIES: CategoryGroup[] = [
  {
    name: 'private',
    label: 'Private Vehicles',
    color: '#3b82f6',
    classes: ['car', 'motorcycle', 'bicycle'],
  },
  {
    name: 'public',
    label: 'Public Transport',
    color: '#4ecdc4',
    classes: ['bus', 'truck', 'train'],
  },
  {
    name: 'people_animals',
    label: 'People & Animals',
    color: '#22c55e',
    classes: ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
  },
  {
    name: 'traffic',
    label: 'Traffic Infrastructure',
    color: '#e85454',
    classes: ['traffic light', 'stop sign', 'fire hydrant', 'parking meter'],
  },
  {
    name: 'objects',
    label: 'Other Objects',
    color: '#9b7bd4',
    classes: [], // catch-all
  },
];

const CategoryBreakdown: React.FC<Props> = ({ detectionData }) => {
  const totals = useMemo(() => {
    if (!detectionData?.session_counts) return [];
    const counts = detectionData.session_counts;
    const matched = new Set<string>();

    const groups = CATEGORIES.map((cat) => {
      const entries = cat.classes
        .map((cls) => [cls, counts[cls] || 0] as const)
        .filter(([, v]) => v > 0);
      entries.forEach(([k]) => matched.add(k));
      const total = entries.reduce((sum, [, v]) => sum + v, 0);
      return { ...cat, total, entries };
    });

    const unmatched = Object.entries(counts)
      .filter(([k]) => !matched.has(k))
      .reduce((sum, [, v]) => sum + v, 0);
    if (unmatched > 0) {
      const objGroup = groups.find((g) => g.name === 'objects')!;
      objGroup.total = unmatched;
    }

    return groups.filter((g) => g.total > 0);
  }, [detectionData?.session_counts]);

  const maxTotal = Math.max(1, ...totals.map((t) => t.total));
  const grandTotal = totals.reduce((sum, t) => sum + t.total, 0);

  return (
    <div className="category-breakdown">
      <div className="panel-header">
        <h3>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="7" height="7" />
            <rect x="14" y="3" width="7" height="7" />
            <rect x="3" y="14" width="7" height="7" />
            <rect x="14" y="14" width="7" height="7" />
          </svg>
          Categories
        </h3>
        {grandTotal > 0 && (
          <span className="total-badge">{grandTotal} total</span>
        )}
      </div>

      <div className="category-list">
        {totals.length === 0 ? (
          <div className="counter-empty">
            <p>No categories yet</p>
          </div>
        ) : (
          totals.map((cat) => {
            const pct = grandTotal > 0 ? Math.round((cat.total / grandTotal) * 100) : 0;
            const barWidth = (cat.total / maxTotal) * 100;

            return (
              <div key={cat.name} className="category-item">
                <div className="category-top">
                  <span className="category-dot" style={{ backgroundColor: cat.color }} />
                  <span className="category-name">{cat.label}</span>
                  <span className="category-pct" style={{ color: cat.color }}>{pct}%</span>
                  <span className="category-count">{cat.total}</span>
                </div>
                <div className="category-bar-track">
                  <div
                    className="category-bar-fill"
                    style={{ width: `${barWidth}%`, backgroundColor: cat.color }}
                  />
                </div>
                {cat.entries.length > 0 && (
                  <div className="category-classes">
                    {cat.entries.map(([cls, cnt]) => (
                      <span key={cls} className="category-class-chip">
                        {cls}: {cnt}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default CategoryBreakdown;
