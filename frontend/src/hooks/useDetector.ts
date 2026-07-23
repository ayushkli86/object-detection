import { useEffect, useRef, useCallback, useState } from 'react';
import type { DetectionData, DetectorStats, ModelsResponse, CameraSource } from '../types';

const API_BASE = `/api`;

/**
 * FPS Optimization:
 * - captureCanvas is 640x480 (matches YOLO imgsz, smaller payload)
 * - Lightweight 8-point pixel hash instead of full getImageData scan
 * - JPEG quality 0.7 (smaller payload, barely noticeable)
 * - Fire-and-forget frames (don't await each response)
 * - requestAnimationFrame scheduling for smoother frame pacing
 */
const CAPTURE_W = 640;
const CAPTURE_H = 480;
const JPEG_QUALITY = 0.7;

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
  remoteFrameUrl: string | null;
  lanIp: string;
  connected: boolean;
  fetchStats: () => Promise<DetectorStats | null>;
  fetchModels: () => Promise<ModelsResponse | null>;
  switchModel: (model: string) => Promise<boolean>;
  activeCameraId: string;
  setActiveCamera: (id: string) => Promise<boolean>;
  availableCameras: CameraSource[];
  fetchCameras: () => Promise<CameraSource[]>;
  updateFps: (fps: number) => void;
  currentFps: number;
}

export function useDetector(): UseDetectorReturn {
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [detectionData, setDetectionData] = useState<DetectionData | null>(null);
  const [frameCanvas, setFrameCanvas] = useState<HTMLCanvasElement | null>(null);
  const [remoteFrameUrl, setRemoteFrameUrl] = useState<string | null>(null);
  const [lanIp, setLanIp] = useState('localhost');
  const [connected, setConnected] = useState(false);
  const [activeCameraId, setActiveCameraId] = useState<string>('local');
  const [availableCameras, setAvailableCameras] = useState<CameraSource[]>([]);
  const [currentFps, setCurrentFps] = useState(30);

  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectingRef = useRef(false);
  const frameTimerRef = useRef<number>(0);
  const prevFrameHashRef = useRef<number>(0);
  const activeCameraIdRef = useRef<string>('local');
  const remotePollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevRemoteUrlRef = useRef<string | null>(null);

  // Keep refs in sync
  useEffect(() => { detectingRef.current = detecting; }, [detecting]);
  useEffect(() => { activeCameraIdRef.current = activeCameraId; }, [activeCameraId]);

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
      video.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none;z-index:-1;';
      document.body.appendChild(video);
      await video.play();

      const canvas = document.createElement('canvas');
      canvas.width = CAPTURE_W;
      canvas.height = CAPTURE_H;

      streamRef.current = stream;
      videoRef.current = video;
      captureCanvasRef.current = canvas;
      setVideoElement(video);
      setFrameCanvas(canvas);
      setCameraState('active');
      return true;
    } catch (err: any) {
      if (videoRef.current?.parentNode) {
        videoRef.current.parentNode.removeChild(videoRef.current);
      }
      videoRef.current = null;
      captureCanvasRef.current = null;
      streamRef.current?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      const msg = err.name === 'NotAllowedError' ? 'Camera permission denied.'
        : err.name === 'NotFoundError' ? 'No camera found on this device.'
        : `Camera error: ${err.message || 'Unknown error'}`;
      setCameraError(msg);
      setCameraState('error');
      return false;
    }
  }, []);

  const cleanupCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current?.parentNode) videoRef.current.parentNode.removeChild(videoRef.current);
    videoRef.current = null;
    captureCanvasRef.current = null;
    setVideoElement(null);
    setFrameCanvas(null);
    setCameraState('idle');
  }, []);

  // ── Local camera frame loop (optimized for FPS) ─────────────────────
  const fpsRef = useRef(30);       // target FPS (can be updated via config)
  const lastSendRef = useRef(0);   // timestamp of last frame send
  const inFlightRef = useRef(false);

  const sendFrameLoop = useCallback(async () => {
    if (!detectingRef.current || activeCameraIdRef.current !== 'local') return;

    // Time-based gate: only send if enough time has elapsed
    const now = performance.now();
    const minInterval = 1000 / fpsRef.current;
    if (now - lastSendRef.current < minInterval) {
      frameTimerRef.current = requestAnimationFrame(sendFrameLoop);
      return;
    }

    // Skip if previous frame is still in flight (backpressure)
    if (inFlightRef.current) {
      frameTimerRef.current = requestAnimationFrame(sendFrameLoop);
      return;
    }

    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return;

    try {
      const ctx = canvas.getContext('2d', { willReadFrequently: false });
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // ── Lightweight 8-point hash (replaces full getImageData scan) ────
      const pts = [
        [0, 0], [canvas.width >> 1, 0], [canvas.width - 1, 0],
        [0, canvas.height - 1], [canvas.width - 1, canvas.height - 1],
        [canvas.width >> 1, canvas.height >> 1],
        [canvas.width >> 2, canvas.height >> 2],
        [(canvas.width * 3) >> 2, (canvas.height * 3) >> 2],
      ];
      let hash = 0;
      for (const [x, y] of pts) {
        const d = ctx.getImageData(x, y, 1, 1).data;
        hash = ((hash << 5) - hash + d[0] + (d[1] << 1) + (d[2] << 2)) | 0;
      }
      if (hash === prevFrameHashRef.current) {
        frameTimerRef.current = requestAnimationFrame(sendFrameLoop);
        return;
      }
      prevFrameHashRef.current = hash;

      // ── Encode + send (fire-and-forget) ─────────────────────────────
      lastSendRef.current = performance.now();
      const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
      inFlightRef.current = true;
      fetch(`${API_BASE}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      })
        .then(res => res.ok ? res.json() : null)
        .then(result => {
          if (result && detectingRef.current && activeCameraIdRef.current === 'local' && result.type === 'detection') {
            setDetectionData(result);
          }
        })
        .catch(() => {})
        .finally(() => { inFlightRef.current = false; });
    } catch (e) {
      inFlightRef.current = false;
    }

    if (detectingRef.current && activeCameraIdRef.current === 'local') {
      frameTimerRef.current = requestAnimationFrame(sendFrameLoop);
    }
  }, []);

  // Allow external FPS updates
  const updateFps = useCallback((fps: number) => {
    const clamped = Math.max(1, Math.min(120, fps));
    fpsRef.current = clamped;
    setCurrentFps(clamped);
  }, []);

  // ── Remote camera functions ──────────────────────────────────────────
  const fetchCameras = useCallback(async (): Promise<CameraSource[]> => {
    try {
      const res = await fetch(`${API_BASE}/cameras`);
      if (!res.ok) return [];
      const data = await res.json();
      const cams = data.cameras || [];
      setAvailableCameras(cams);
      return cams;
    } catch { return []; }
  }, []);

  const stopRemotePoll = useCallback(() => {
    if (remotePollRef.current) {
      clearTimeout(remotePollRef.current);
      remotePollRef.current = null;
    }
    if (prevRemoteUrlRef.current) {
      URL.revokeObjectURL(prevRemoteUrlRef.current);
      prevRemoteUrlRef.current = null;
    }
    setRemoteFrameUrl(null);
  }, []);

  const startRemotePoll = useCallback(async () => {
    if (!detectingRef.current || activeCameraIdRef.current === 'local') return;
    try {
      const res = await fetch(`${API_BASE}/remote-frame`);
      if (res.ok) {
        const data = await res.json();
        if (detectingRef.current && activeCameraIdRef.current !== 'local' && data.type === 'detection' && data.image) {
          setDetectionData(data as DetectionData);
          const jpegBase64 = data.image.split(',')[1];
          if (jpegBase64) {
            const byteStr = atob(jpegBase64);
            const ab = new ArrayBuffer(byteStr.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteStr.length; i++) ia[i] = byteStr.charCodeAt(i);
            const blob = new Blob([ab], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            if (prevRemoteUrlRef.current) URL.revokeObjectURL(prevRemoteUrlRef.current);
            prevRemoteUrlRef.current = url;
            setRemoteFrameUrl(url);
          }
        }
      }
    } catch (e) {
      console.warn('Remote frame poll failed:', e);
    }
    if (detectingRef.current && activeCameraIdRef.current !== 'local') {
      remotePollRef.current = setTimeout(startRemotePoll, 100);
    }
  }, []);

  const setActiveCamera = useCallback(async (id: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/cameras/select`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: id }),
      });
      if (!res.ok) return false;

      const wasDetecting = detectingRef.current;
      const wasLocal = activeCameraIdRef.current === 'local';
      const nowRemote = id !== 'local';

      // Stop old source
      if (wasDetecting && wasLocal) {
        if (frameTimerRef.current) {
          cancelAnimationFrame(frameTimerRef.current);
          frameTimerRef.current = 0;
        }
      }
      if (wasDetecting && !wasLocal) {
        stopRemotePoll();
      }

      // Switch
      setActiveCameraId(id);
      setDetectionData(null);
      setRemoteFrameUrl(null);

      if (wasDetecting && nowRemote) {
        setTimeout(() => startRemotePoll(), 100);
      } else if (wasDetecting && !nowRemote) {
        setTimeout(() => sendFrameLoop(), 100);
      } else if (nowRemote) {
        // Auto-start detection when switching to remote camera
        detectingRef.current = true;
        setDetecting(true);
        setTimeout(() => startRemotePoll(), 100);
      }

      return true;
    } catch { return false; }
  }, [stopRemotePoll, startRemotePoll, sendFrameLoop]);

  // ── Start / Stop ─────────────────────────────────────────────────────
  const startDetection = useCallback(() => {
    detectingRef.current = true;
    setDetecting(true);
    if (activeCameraIdRef.current === 'local') {
      if (!streamRef.current || cameraState !== 'active') {
        detectingRef.current = false;
        setDetecting(false);
        return;
      }
      sendFrameLoop();
    } else {
      startRemotePoll();
    }
  }, [cameraState, sendFrameLoop, startRemotePoll]);

  const stopDetection = useCallback(() => {
    detectingRef.current = false;
    setDetecting(false);
    if (frameTimerRef.current) {
      cancelAnimationFrame(frameTimerRef.current);
      frameTimerRef.current = 0;
    }
    inFlightRef.current = false;
    stopRemotePoll();
    setDetectionData(null);
  }, [stopRemotePoll]);

  // ── Stats / Models ───────────────────────────────────────────────────
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

  // ── Health + camera poll ─────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const [healthRes, statsRes] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/stats`),
        ]);
        const healthOk = healthRes.ok;
        setConnected(healthOk);
        if (healthOk) {
          const healthData = await healthRes.json();
          if (healthData.lan_ip) setLanIp(healthData.lan_ip);
        }
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          if (statsData.max_fps) {
            fpsRef.current = Math.max(1, Math.min(120, statsData.max_fps));
            setCurrentFps(fpsRef.current);
          }
        }
      } catch { setConnected(false); }
    };
    check();
    const interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!connected) return;
    let cancelled = false;
    const poll = async () => { await fetchCameras(); };
    poll();
    const interval = setInterval(poll, 2000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [connected, fetchCameras]);

  // ── Cleanup ──────────────────────────────────────────────────────────
  useEffect(() => {
    return () => { stopDetection(); cleanupCamera(); };
  }, [stopDetection, cleanupCamera]);

  return {
    cameraState, cameraError, requestCamera, videoElement,
    startDetection, stopDetection, detecting,
    detectionData, frameCanvas, remoteFrameUrl, lanIp,
    connected, fetchStats, fetchModels, switchModel,
    activeCameraId, setActiveCamera, availableCameras, fetchCameras,
    updateFps, currentFps,
  };
}
