import { useEffect, useRef, useCallback, useState } from 'react';
import type { DetectionData, DetectorStats, ModelsResponse, CameraSource } from '../types';

const API_BASE = `/api`;
const FRAME_INTERVAL_MS = 100;

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

  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectingRef = useRef(false);
  const frameTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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

  // ── Local camera frame loop ──────────────────────────────────────────
  const sendFrameLoop = useCallback(async () => {
    if (!detectingRef.current || activeCameraIdRef.current !== 'local') return;
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return;

    try {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

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
        if (detectingRef.current && activeCameraIdRef.current === 'local' && result.type === 'detection') {
          setDetectionData(result);
        }
      }
    } catch (e) {
      console.warn('Detection request failed:', e);
    }

    if (detectingRef.current && activeCameraIdRef.current === 'local') {
      frameTimerRef.current = setTimeout(sendFrameLoop, FRAME_INTERVAL_MS);
    }
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
      remotePollRef.current = setTimeout(startRemotePoll, FRAME_INTERVAL_MS);
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
          clearTimeout(frameTimerRef.current);
          frameTimerRef.current = null;
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
      clearTimeout(frameTimerRef.current);
      frameTimerRef.current = null;
    }
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
        const res = await fetch(`${API_BASE}/health`);
        setConnected(res.ok);
        if (res.ok) {
          const data = await res.json();
          if (data.lan_ip) setLanIp(data.lan_ip);
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
  };
}
