"use client";

import { useState, useCallback } from "react";
import { Upload, Image as ImageIcon, Loader2, AlertCircle } from "lucide-react";
import { apiPost } from "@/lib/api";
import { ImageDetectionResponse } from "@/lib/types";

export default function ImageDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ImageDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modules, setModules] = useState<string>("");
  const [confidence, setConfidence] = useState(0.45);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(selected);
      setResult(null);
      setError(null);
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("modules", modules);
      formData.append("confidence", confidence.toString());
      const data = await apiPost("/api/detect/image", formData);
      setResult(data as ImageDetectionResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }, [file, modules, confidence]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">Image Detection</h1>
        <p className="text-dark-200">Upload an image for AI object detection</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Upload Panel */}
        <div className="space-y-4">
          <div className="rounded-xl border border-dashed border-dark-500 bg-dark-800 p-8 text-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="image-upload"
            />
            <label htmlFor="image-upload" className="cursor-pointer">
              <Upload className="mx-auto h-12 w-12 text-dark-200" />
              <p className="mt-4 text-dark-50">Click to upload an image</p>
              <p className="text-sm text-dark-200">PNG, JPG, WEBP up to 50MB</p>
            </label>
          </div>

          {preview && (
            <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
              <img
                src={preview}
                alt="Preview"
                className="mx-auto max-h-64 rounded-lg"
              />
            </div>
          )}

          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-sm text-dark-200">Modules (comma-separated)</label>
              <input
                type="text"
                value={modules}
                onChange={(e) => setModules(e.target.value)}
                placeholder="objects, face_recognition, anpr"
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
                <ImageIcon className="h-5 w-5" />
              )}
              <span>{loading ? "Processing..." : "Run Detection"}</span>
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

          {result && (
            <div className="space-y-4">
              <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-dark-50">Results</h3>
                  <span className="text-sm text-dark-200">
                    {result.processing_time_ms.toFixed(0)}ms
                  </span>
                </div>
              </div>

              {result.annotated_image_base64 && (
                <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                    alt="Result"
                    className="w-full rounded-lg"
                  />
                </div>
              )}

              {result.detections.length > 0 && (
                <div className="rounded-xl border border-dark-500 bg-dark-800">
                  <div className="border-b border-dark-500 px-4 py-3">
                    <h3 className="font-semibold text-dark-50">
                      Detections ({result.detections.length})
                    </h3>
                  </div>
                  <div className="max-h-64 overflow-y-auto p-4">
                    <div className="space-y-2">
                      {result.detections.map((det, idx) => (
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

              {result.detections.length === 0 && (
                <div className="rounded-xl border border-dark-500 bg-dark-800 p-8 text-center">
                  <p className="text-dark-200">No objects detected</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
