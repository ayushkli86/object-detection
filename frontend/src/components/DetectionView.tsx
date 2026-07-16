import React, { useRef, useEffect, useState } from 'react';
import type { DetectionData } from '../types';

interface Props {
  videoElement: HTMLVideoElement | null;
  annotatedFrameUrl: string | null;
  detecting: boolean;
  detectionData: DetectionData | null;
}

const DetectionView: React.FC<Props> = ({
  videoElement,
  annotatedFrameUrl,
  detecting,
  detectionData,
}) => {
  const liveCanvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  // ── Live camera preview (draws video to canvas each frame) ──────────
  useEffect(() => {
    if (!videoElement || detecting) {
      cancelAnimationFrame(animFrameRef.current);
      return;
    }

    const canvas = liveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      if (detecting) return;
      if (videoElement.readyState >= 2) {
        canvas.width = videoElement.videoWidth || 640;
        canvas.height = videoElement.videoHeight || 480;
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      }
      animFrameRef.current = requestAnimationFrame(draw);
    };
    draw();

    return () => cancelAnimationFrame(animFrameRef.current);
  }, [videoElement, detecting]);

  // ── When detecting, show annotated frame (image from backend) ────────
  if (detecting && annotatedFrameUrl) {
    return (
      <div className="detection-view">
        <div className="frame-container">
          <img
            src={annotatedFrameUrl}
            alt="Detection"
            className="frame-image"
          />

          {detectionData && (
            <div className="frame-overlay">
              <span className="badge badge-source">CAMERA</span>
              <span className="badge badge-fps">{detectionData.fps.toFixed(1)} FPS</span>
              <span className="badge badge-objects">
                {detectionData.active_tracks_count} active · {detectionData.total_unique_seen} unique
              </span>
              {detectionData.inference_ms > 0 && (
                <span className="badge badge-latency">
                  {detectionData.inference_ms.toFixed(0)}ms infer
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Idle: live camera preview ────────────────────────────────────────
  return (
    <div className="detection-view">
      <div className="frame-container">
        <canvas ref={liveCanvasRef} className="frame-image" />
        <div className="frame-overlay">
          <span className="badge badge-source">CAMERA PREVIEW</span>
        </div>
      </div>
    </div>
  );
};

export default DetectionView;
