"use client";

import { useState, useCallback } from "react";
import { Upload, Video, Loader2, AlertCircle, Play, Square } from "lucide-react";
import { apiPost } from "@/lib/api";
import { VideoDetectionResponse } from "@/lib/types";

export default function VideoDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VideoDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modules, setModules] = useState<string>("");
  const [confidence, setConfidence] = useState(0.35);
  const [progress, setProgress] = useState(0);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setResult(null);
      setError(null);
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("modules", modules);
      formData.append("confidence", confidence.toString());

      // Simulate progress for UX
      const progressInterval = setInterval(() => {
        setProgress((p) => Math.min(p + 5, 90));
      }, 500);

      const data = await apiPost("/api/detect/video", formData);
      clearInterval(progressInterval);
      setProgress(100);
      setResult(data as VideoDetectionResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }, [file, modules, confidence]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">Video Detection</h1>
        <p className="text-dark-200">Upload a video for AI detection analysis</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Upload Panel */}
        <div className="space-y-4">
          <div className="rounded-xl border border-dashed border-dark-500 bg-dark-800 p-8 text-center">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="hidden"
              id="video-upload"
            />
            <label htmlFor="video-upload" className="cursor-pointer">
              <Video className="mx-auto h-12 w-12 text-dark-200" />
              <p className="mt-4 text-dark-50">Click to upload a video</p>
              <p className="text-sm text-dark-200">MP4, AVI, MOV up to 50MB</p>
            </label>
          </div>

          {file && (
            <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
              <p className="text-sm text-dark-200">Selected: {file.name}</p>
              <p className="text-xs text-dark-200">
                {(file.size / (1024 * 1024)).toFixed(1)} MB
              </p>
            </div>
          )}

          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-sm text-dark-200">Modules (comma-separated)</label>
              <input
                type="text"
                value={modules}
                onChange={(e) => setModules(e.target.value)}
                placeholder="objects, vehicle, anpr"
                className="w-full rounded-lg border border-dark-500 bg-dark-800 px-3 py-2 text-sm text-dark-50 focus:border-primary-400 focus:outline-none"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm text-dark-200">Confidence: {confidence.toFixed(2)}</label>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full accent-primary-400"
              />
            </div>
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="flex w-full items-center justify-center space-x-2 rounded-lg bg-primary-600 py-3 text-sm font-medium text-white hover:bg-primary-500 disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Play className="h-5 w-5" />
              )}
              <span>{loading ? "Processing..." : "Process Video"}</span>
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div className="space-y-4">
          {error && (
            <div className="flex items-center space-x-2 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          )}

          {loading && (
            <div className="rounded-xl border border-dark-500 bg-dark-800 p-8">
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <Loader2 className="h-6 w-6 animate-spin text-primary-400" />
                  <span className="text-dark-50">Processing video...</span>
                </div>
                <div className="h-2 w-full rounded-full bg-dark-700">
                  <div
                    className="h-2 rounded-full bg-primary-400 transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-sm text-dark-200">{progress}%</p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-dark-50">Processing Complete</h3>
                  <span className="text-sm text-dark-200">
                    {result.processing_time_ms.toFixed(0)}ms
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-dark-200">Total Frames</p>
                    <p className="font-semibold text-dark-50">{result.total_frames}</p>
                  </div>
                  <div>
                    <p className="text-dark-200">Detections</p>
                    <p className="font-semibold text-dark-50">{result.detections.length}</p>
                  </div>
                </div>
              </div>

              {result.detections.length > 0 && (
                <div className="rounded-xl border border-dark-500 bg-dark-800">
                  <div className="border-b border-dark-500 px-4 py-3">
                    <h3 className="font-semibold text-dark-50">
                      Detections ({result.detections.length})
                    </h3>
                  </div>
                  <div className="max-h-64 overflow-y-auto p-4">
                    <div className="space-y-2">
                      {result.detections.slice(0, 50).map((det, idx) => (
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
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
