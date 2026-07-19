import { useEffect, useRef, useCallback, useState } from 'react';
import type { DetectionData, DetectorStats, ModelsResponse } from '../types';

const API_BASE = `/api`;
const FRAME_INTERVAL_MS = 100; // ~10 FPS

export type CameraState = 'idle' | 'requesting' | 'active' | 'error';

export interface UseDetectorReturn {
  cameraState: CameraState;
  cameraError: string | null;
  requestCamera: () => Promise<boolean>;
  videoElement: HTMLVideoElement | null;
  startDetection: () => void;
  stopDetection: () => void;
  detecting: boolean;
  detectionData: DetectionData | null;
  frameCanvas: HTMLCanvasElement | null;
  connected: boolean;
  fetchStats: () => Promise<DetectorStats | null>;
  fetchModels: () => Promise<ModelsResponse | null>;
  switchModel: (model: string) => Promise<boolean>;
}

export function useDetector(): UseDetectorReturn {
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [detectionData, setDetectionData] = useState<DetectionData | null>(null);
  const [frameCanvas, setFrameCanvas] = useState<HTMLCanvasElement | null>(null);
  const [connected, setConnected] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectingRef = useRef(false);
  const frameTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevFrameHashRef = useRef<number>(0);

  // ── Camera ───────────────────────────────────────────────────────────
  const requestCamera = useCallback(async (): Promise<boolean> => {
    setCameraState('requesting');
    setCameraError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      video.muted = true;
      video.style.cssText =
        'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none;z-index:-1;';
      document.body.appendChild(video);
      await video.play();

      const canvas = document.createElement('canvas');
      canvas.width = 960;
      canvas.height = 720;

      streamRef.current = stream;
      videoRef.current = video;
      captureCanvasRef.current = canvas;
      setVideoElement(video);
      setFrameCanvas(canvas);
      setCameraState('active');
      return true;
    } catch (err: any) {
      const msg =
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access and reload.'
          : err.name === 'NotFoundError'
            ? 'No camera found on this device.'
            : `Camera error: ${err.message || 'Unknown error'}`;
      setCameraError(msg);
      setCameraState('error');
      return false;
    }
  }, []);

  const cleanupCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current?.parentNode) {
      videoRef.current.parentNode.removeChild(videoRef.current);
    }
    videoRef.current = null;
    captureCanvasRef.current = null;
    setVideoElement(null);
    setFrameCanvas(null);
    setCameraState('idle');
  }, []);

  // ── Detection frame loop ─────────────────────────────────────────────
  const sendFrameLoop = useCallback(async () => {
    if (!detectingRef.current) return;

    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return;

    try {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Frame dedup
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let hash = 0;
      for (let i = 0; i < imgData.data.length; i += 64) {
        hash = ((hash << 5) - hash + imgData.data[i]) | 0;
      }
      if (hash === prevFrameHashRef.current) {
        frameTimerRef.current = setTimeout(sendFrameLoop, FRAME_INTERVAL_MS / 2);
        return;
      }
      prevFrameHashRef.current = hash;

      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      const res = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });

      if (res.ok) {
        const result: DetectionData = await res.json();
        if (detectingRef.current && result.type === 'detection') {
          setDetectionData(result);
        }
      }
    } catch {
      // Skip frame on network error
    }

    if (detectingRef.current) {
      frameTimerRef.current = setTimeout(sendFrameLoop, FRAME_INTERVAL_MS);
    }
  }, []);

  // ── Start / Stop ─────────────────────────────────────────────────────
  const startDetection = useCallback(() => {
    if (!streamRef.current || cameraState !== 'active') return;
    detectingRef.current = true;
    setDetecting(true);
    sendFrameLoop();
  }, [cameraState, sendFrameLoop]);

  const stopDetection = useCallback(() => {
    detectingRef.current = false;
    setDetecting(false);
    if (frameTimerRef.current) {
      clearTimeout(frameTimerRef.current);
      frameTimerRef.current = null;
    }
    setDetectionData(null);
  }, []);

  // ── Stats ────────────────────────────────────────────────────────────
  const fetchStats = useCallback(async (): Promise<DetectorStats | null> => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (!res.ok) return null;
      return await res.json();
    } catch { return null; }
  }, []);

  const fetchModels = useCallback(async (): Promise<ModelsResponse | null> => {
    try {
      const res = await fetch(`${API_BASE}/models`);
      if (!res.ok) return null;
      return await res.json();
    } catch { return null; }
  }, []);

  const switchModel = useCallback(async (model: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/model`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      });
      return res.ok;
    } catch { return false; }
  }, []);

  // ── Health check ─────────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        setConnected(res.ok);
      } catch { setConnected(false); }
    };
    check();
    const interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);

  // ── Cleanup ─────────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopDetection();
      cleanupCamera();
    };
  }, [stopDetection, cleanupCamera]);

  return {
    cameraState, cameraError, requestCamera, videoElement,
    startDetection, stopDetection, detecting,
    detectionData, frameCanvas,
    connected,
    fetchStats, fetchModels,
    switchModel,
  };
}
