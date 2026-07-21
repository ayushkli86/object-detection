import React, { useMemo } from 'react';
import type { DetectionData } from '../types';

interface Props {
  detectionData: DetectionData | null;
}

/**
 * Dynamic category breakdown — works with ANY model (COCO 80, Open Images 601, etc.)
 * Categories are built from the actual class names in detection data,
 * not hardcoded COCO names.
 */

// Keywords to classify detected objects into categories
const CATEGORY_RULES: { name: string; label: string; color: string; keywords: string[] }[] = [
  {
    name: 'vehicles',
    label: 'Vehicles',
    color: '#3b82f6',
    keywords: ['car', 'motorcycle', 'motorbike', 'bicycle', 'bike', 'bus', 'truck', 'train', 'boat', 'airplane', 'vehicle', 'tractor', 'auto', 'rickshaw', 'tempo', 'van', 'suv', 'sedan', 'taxi', 'ambulance', 'fire truck', 'police'],
  },
  {
    name: 'people',
    label: 'People',
    color: '#22c55e',
    keywords: ['person', 'man', 'woman', 'child', 'boy', 'girl', 'pedestrian', 'rider'],
  },
  {
    name: 'animals',
    label: 'Animals',
    color: '#c850c0',
    keywords: ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'chicken', 'rabbit', 'animal', 'pigeon', 'monkey'],
  },
  {
    name: 'infrastructure',
    label: 'Traffic Infrastructure',
    color: '#e85454',
    keywords: ['traffic light', 'stop sign', 'fire hydrant', 'parking meter', 'traffic sign', 'street sign', 'cone', 'barrier', 'zebra crossing', 'speed bump', 'crosswalk'],
  },
  {
    name: 'furniture',
    label: 'Furniture',
    color: '#82A2E6',
    keywords: ['chair', 'table', 'desk', 'couch', 'sofa', 'bed', 'shelf', 'cabinet', 'bench', 'stool', 'wardrobe'],
  },
  {
    name: 'electronics',
    label: 'Electronics',
    color: '#FFC882',
    keywords: ['laptop', 'computer', 'keyboard', 'mouse', 'monitor', 'cell phone', 'telephone', 'tv', 'television', 'remote', 'printer', 'screen', 'desktop', 'tablet', 'camera', 'speaker'],
  },
  {
    name: 'kitchen',
    label: 'Kitchen',
    color: '#FF8282',
    keywords: ['bottle', 'cup', 'glass', 'bowl', 'plate', 'fork', 'knife', 'spoon', 'refrigerator', 'microwave', 'oven', 'toaster', 'sink', 'pot', 'pan'],
  },
  {
    name: 'personal',
    label: 'Personal Items',
    color: '#F79646',
    keywords: ['backpack', 'handbag', 'suitcase', 'umbrella', 'wallet', 'purse', 'bag', 'hat', 'shoe', 'boot', 'sandal'],
  },
  {
    name: 'stationery',
    label: 'Stationery',
    color: '#FFFF82',
    keywords: ['book', 'pen', 'pencil', 'notebook', 'paper', 'scissors', 'ruler', 'marker', 'eraser', 'folder', 'binder'],
  },
  {
    name: 'food',
    label: 'Food',
    color: '#6bcb77',
    keywords: ['banana', 'apple', 'orange', 'sandwich', 'pizza', 'cake', 'donut', 'hot dog', 'broccoli', 'carrot', 'fruit', 'vegetable', 'bread'],
  },
  {
    name: 'sports',
    label: 'Sports',
    color: '#4ecdc4',
    keywords: ['sports ball', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis racket', 'frisbee', 'skis', 'snowboard', 'kite', 'racket'],
  },
  {
    name: 'clothing',
    label: 'Clothing',
    color: '#a78bfa',
    keywords: ['shirt', 'jacket', 'pants', 'dress', 'hat', 'shoe', 'tie', 'coat', 'clothing', 'clothes', 'helmet'],
  },
  {
    name: 'tools',
    label: 'Tools',
    color: '#f59e0b',
    keywords: ['hammer', 'screwdriver', 'wrench', 'drill', 'tool', 'saw'],
  },
];

function classifyObject(name: string): { label: string; color: string } {
  const lower = name.toLowerCase().trim();
  for (const rule of CATEGORY_RULES) {
    for (const kw of rule.keywords) {
      if (lower === kw || lower.includes(kw) || kw.includes(lower)) {
        return { label: rule.label, color: rule.color };
      }
    }
  }
  return { label: 'Other Objects', color: '#64748b' };
}

const CategoryBreakdown: React.FC<Props> = ({ detectionData }) => {
  const { totals, grandTotal } = useMemo(() => {
    if (!detectionData?.session_counts) return { totals: [], grandTotal: 0 };
    const counts = detectionData.session_counts;

    // Dynamically group all detected classes into categories
    const categoryMap = new Map<string, { label: string; color: string; total: number; entries: [string, number][] }>();

    for (const [className, count] of Object.entries(counts)) {
      if (count <= 0) continue;
      const { label, color } = classifyObject(className);
      const key = label;
      if (!categoryMap.has(key)) {
        categoryMap.set(key, { label, color, total: 0, entries: [] });
      }
      const cat = categoryMap.get(key)!;
      cat.total += count;
      cat.entries.push([className, count]);
    }

    const sorted = Array.from(categoryMap.values())
      .sort((a, b) => b.total - a.total);

    const grand = sorted.reduce((sum, t) => sum + t.total, 0);
    return { totals: sorted, grandTotal: grand };
  }, [detectionData?.session_counts]);

  const maxTotal = Math.max(1, ...totals.map((t) => t.total));

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
              <div key={cat.label} className="category-item">
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
                    {cat.entries.slice(0, 8).map(([cls, cnt]) => (
                      <span key={cls} className="category-class-chip">
                        {cls}: {cnt}
                      </span>
                    ))}
                    {cat.entries.length > 8 && (
                      <span className="category-class-chip" style={{ opacity: 0.5 }}>
                        +{cat.entries.length - 8} more
                      </span>
                    )}
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
