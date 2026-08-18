"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { Camera, Square, Play, AlertCircle } from "lucide-react";
import { apiGet, getWsUrl } from "@/lib/api";

interface DetectionResult {
  label: string;
  confidence: number;
  box: number[];
  source: string;
}

export default function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [detections, setDetections] = useState<DetectionResult[]>([]);
  const [fps, setFps] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedModules, setSelectedModules] = useState<string[]>(["objects"]);
  const [confidence, setConfidence] = useState(0.45);
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);

  useEffect(() => {
    async function loadCameras() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        setCameras(videoDevices);
        if (videoDevices.length > 0 && !cameraId) {
          setCameraId(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error("Failed to enumerate cameras:", err);
      }
    }
    loadCameras();
  }, [cameraId]);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const constraints: MediaTrackConstraints = {
        deviceId: cameraId ? { exact: cameraId } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "environment",
      };
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: constraints,
        audio: false,
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsRunning(true);
    } catch (err) {
      console.error("Camera error:", err);
      setError(err instanceof Error ? err.message : "Camera permission denied or unavailable");
    }
  }, [cameraId]);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsRunning(false);
    setDetections([]);
  }, [stream]);

  useEffect(() => {
    if (!isRunning) return;

    const video = videoRef.current;
    if (!video) return;

    const handlePlay = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      let lastTime = performance.now();
      let frameCount = 0;

      const processFrame = async () => {
        if (!isRunning || video.paused || video.ended) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Capture frame for WebSocket processing
        const frameData = canvas.toDataURL("image/jpeg", 0.8);

        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            frame_base64: frameData,
            modules: selectedModules,
            confidence: confidence,
          }));
        }

        // Calculate FPS
        frameCount++;
        const now = performance.now();
        if (now - lastTime >= 1000) {
          setFps(frameCount);
          frameCount = 0;
          lastTime = now;
        }

        requestAnimationFrame(processFrame);
      };

      processFrame();
    };

    video.addEventListener("play", handlePlay);
    return () => video.removeEventListener("play", handlePlay);
  }, [isRunning, selectedModules, confidence]);

  useEffect(() => {
    if (!isRunning) return;

    const ws = new WebSocket(getWsUrl("/ws/detection"));
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setDetections(data.detections || []);
        setFps(data.fps || fps);
        setProcessingTime(data.processing_time_ms || 0);

        // Draw detections on canvas
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (canvas && video && data.annotated_image_base64) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            const img = new Image();
            img.onload = () => {
              canvas.width = img.width;
              canvas.height = img.height;
              ctx.drawImage(img, 0, 0);
            };
            img.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
          }
        }
      } catch (err) {
        console.error("WebSocket message error:", err);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
    };

    return () => {
      ws.close();
    };
  }, [isRunning]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-dark-50">Live Camera</h1>
          <p className="text-dark-200">Real-time AI detection from your webcam</p>
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value)}
            className="h-10 rounded-lg border border-dark-500 bg-dark-800 px-3 text-sm text-dark-50 focus:border-primary-400 focus:outline-none"
          >
            {cameras.map((cam) => (
              <option key={cam.deviceId} value={cam.deviceId}>
                {cam.label || `Camera ${cam.deviceId.slice(0, 8)}`}
              </option>
            ))}
          </select>
          {!isRunning ? (
            <button
              onClick={startCamera}
              className="flex items-center space-x-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-500"
            >
              <Play className="h-4 w-4" />
              <span>Start Camera</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="flex items-center space-x-2 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-500"
            >
              <Square className="h-4 w-4" />
              <span>Stop Camera</span>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="flex items-center space-x-2 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400">
          <AlertCircle className="h-5 w-5" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Camera Feed */}
        <div className="lg:col-span-2">
          <div className="relative overflow-hidden rounded-xl border border-dark-500 bg-dark-800">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="hidden"
            />
            <canvas
              ref={canvasRef}
              className="w-full bg-dark-950"
              style={{ aspectRatio: "16/9" }}
            />
            {!isRunning && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Camera className="mx-auto h-16 w-16 text-dark-500" />
                  <p className="mt-4 text-dark-200">Click "Start Camera" to begin</p>
                </div>
              </div>
            )}
            {isRunning && (
              <div className="absolute top-4 left-4 flex items-center space-x-4 rounded-lg bg-dark-900/80 px-3 py-2">
                <span className="text-sm font-mono text-green-400">FPS: {fps}</span>
                <span className="text-sm font-mono text-primary-400">{processingTime.toFixed(0)}ms</span>
              </div>
            )}
          </div>

          {/* Settings */}
          <div className="mt-4 flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-dark-200">Confidence:</label>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="h-2 w-32 accent-primary-400"
              />
              <span className="text-sm text-dark-50">{confidence.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Detections Panel */}
        <div className="space-y-4">
          <div className="rounded-xl border border-dark-500 bg-dark-800">
            <div className="border-b border-dark-500 px-4 py-3">
              <h3 className="font-semibold text-dark-50">Live Detections</h3>
            </div>
            <div className="max-h-96 overflow-y-auto p-4">
              {detections.length === 0 ? (
                <p className="text-sm text-dark-200">No detections yet</p>
              ) : (
                <div className="space-y-2">
                  {detections.map((det, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between rounded-lg bg-dark-700 px-3 py-2"
                    >
                      <span className="text-sm font-medium text-dark-50">{det.label}</span>
                      <span className="text-xs text-dark-200">
                        {(det.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Module Selector */}
          <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
            <h3 className="mb-3 font-semibold text-dark-50">Active Modules</h3>
            <div className="space-y-2">
              {["objects", "face_recognition", "anpr", "qr_codes"].map((mod) => (
                <label key={mod} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedModules.includes(mod)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedModules([...selectedModules, mod]);
                      } else {
                        setSelectedModules(selectedModules.filter(m => m !== mod));
                      }
                    }}
                    className="accent-primary-400"
                  />
                  <span className="text-sm text-dark-200">{mod}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
