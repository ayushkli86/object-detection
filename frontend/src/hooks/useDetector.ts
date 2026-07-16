import { useEffect, useRef, useCallback, useState } from 'react';
import type { DetectionData, DetectorStats } from '../types';

const API_BASE = `/api`;
const FRAME_INTERVAL_MS = 120; // ~8 FPS to backend

export type CameraState = 'idle' | 'requesting' | 'active' | 'error';

export interface UseDetectorReturn {
  // Camera
  cameraState: CameraState;
  cameraError: string | null;
  requestCamera: () => Promise<boolean>;
  videoElement: HTMLVideoElement | null;

  // Detection
  startDetection: () => void;
  stopDetection: () => void;
  detecting: boolean;

  // Detection results
  detectionData: DetectionData | null;
  annotatedFrameUrl: string | null;

  // Server
  connected: boolean;

  // Stats
  fetchStats: () => Promise<DetectorStats | null>;
}

export function useDetector(): UseDetectorReturn {
  // ── Camera state ──────────────────────────────────────────────────────
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);

  // ── Detection state ──────────────────────────────────────────────────
  const [detecting, setDetecting] = useState(false);
  const [detectionData, setDetectionData] = useState<DetectionData | null>(null);
  const [annotatedFrameUrl, setAnnotatedFrameUrl] = useState<string | null>(null);

  // ── Server state ─────────────────────────────────────────────────────
  const [connected, setConnected] = useState(false);

  // ── Refs ─────────────────────────────────────────────────────────────
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectingRef = useRef(false);
  const frameTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevFrameUrlRef = useRef<string | null>(null);

  // ── Camera ───────────────────────────────────────────────────────────
  const requestCamera = useCallback(async (): Promise<boolean> => {
    setCameraState('requesting');
    setCameraError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      // Create video element (must be in DOM for playback)
      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      video.muted = true;
      video.style.cssText =
        'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none;z-index:-1;';
      document.body.appendChild(video);
      await video.play();

      // Create offscreen canvas for frame capture
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;

      streamRef.current = stream;
      videoRef.current = video;
      canvasRef.current = canvas;
      setVideoElement(video);
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

  // ── Cleanup camera ───────────────────────────────────────────────────
  const cleanupCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current?.parentNode) {
      videoRef.current.parentNode.removeChild(videoRef.current);
    }
    videoRef.current = null;
    canvasRef.current = null;
    setVideoElement(null);
    setCameraState('idle');
  }, []);

  // ── Detection frame loop ─────────────────────────────────────────────
  const sendFrameLoop = useCallback(async () => {
    if (!detectingRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    try {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

      const res = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });

      if (res.ok) {
        const result: DetectionData = await res.json();
        if (detectingRef.current && result.type === 'detection') {
          setDetectionData(result);

          // Swap annotated frame URL (avoid leak)
          if (prevFrameUrlRef.current && prevFrameUrlRef.current.startsWith('blob:')) {
            URL.revokeObjectURL(prevFrameUrlRef.current);
          }
          if (result.image) {
            prevFrameUrlRef.current = result.image;
            setAnnotatedFrameUrl(result.image);
          }
        }
      }
    } catch {
      // Skip frame on network error
    }

    if (detectingRef.current) {
      frameTimerRef.current = setTimeout(sendFrameLoop, FRAME_INTERVAL_MS);
    }
  }, []);

  // ── Start / Stop detection ───────────────────────────────────────────
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
    // Clean up annotated frame
    if (prevFrameUrlRef.current && prevFrameUrlRef.current.startsWith('blob:')) {
      URL.revokeObjectURL(prevFrameUrlRef.current);
    }
    prevFrameUrlRef.current = null;
    setAnnotatedFrameUrl(null);
    setDetectionData(null);
  }, []);

  // ── Stats fetchers ───────────────────────────────────────────────────
  const fetchStats = useCallback(async (): Promise<DetectorStats | null> => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }, []);

  // ── Health check on mount ────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) setConnected(true);
        else setConnected(false);
      } catch {
        setConnected(false);
      }
    };
    check();
    const interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);

  // ── Cleanup on unmount ───────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopDetection();
      cleanupCamera();
    };
  }, [stopDetection, cleanupCamera]);

  return {
    cameraState,
    cameraError,
    requestCamera,
    videoElement,
    startDetection,
    stopDetection,
    detecting,
    detectionData,
    annotatedFrameUrl,
    connected,
    fetchStats,
  };
}
