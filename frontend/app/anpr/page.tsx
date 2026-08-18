"use client";

import { useState, useCallback } from "react";
import { Upload, Car, Loader2, AlertCircle } from "lucide-react";
import { apiPost } from "@/lib/api";
import { ANPRResponse } from "@/lib/types";

export default function ANPRPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ANPRResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.35);

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
      formData.append("confidence", confidence.toString());
      const data = await apiPost("/api/anpr/image", formData);
      setResult(data as ANPRResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "ANPR processing failed");
    } finally {
      setLoading(false);
    }
  }, [file, confidence]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">ANPR — Number Plate Recognition</h1>
        <p className="text-dark-200">Detect and read license plates from images</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <div className="rounded-xl border border-dashed border-dark-500 bg-dark-800 p-8 text-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="anpr-upload"
            />
            <label htmlFor="anpr-upload" className="cursor-pointer">
              <Car className="mx-auto h-12 w-12 text-dark-200" />
              <p className="mt-4 text-dark-50">Upload vehicle image</p>
              <p className="text-sm text-dark-200">PNG, JPG, WEBP</p>
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
              <Car className="h-5 w-5" />
            )}
            <span>{loading ? "Processing..." : "Run ANPR"}</span>
          </button>
        </div>

        <div className="space-y-4">
          {error && (
            <div className="flex items-center space-x-2 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {result.annotated_image_base64 && (
                <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                    alt="ANPR Result"
                    className="w-full rounded-lg"
                  />
                </div>
              )}

              <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
                <h3 className="mb-4 font-semibold text-dark-50">Plate Detection Result</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-dark-200">Plate Text</p>
                    <p className="text-2xl font-bold text-primary-400">
                      {result.plate_text || "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-dark-200">Confidence</p>
                    <p className="text-2xl font-bold text-dark-50">
                      {result.confidence ? `${(result.confidence * 100).toFixed(0)}%` : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-dark-200">Vehicle Box</p>
                    <p className="font-mono text-sm text-dark-50">
                      {result.vehicle_box ? result.vehicle_box.join(", ") : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-dark-200">Plate Box</p>
                    <p className="font-mono text-sm text-dark-50">
                      {result.plate_box ? result.plate_box.join(", ") : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-dark-200">Timestamp</p>
                    <p className="text-sm text-dark-50">{result.timestamp || "—"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-dark-200">Processing Time</p>
                    <p className="text-sm text-dark-50">{result.processing_time_ms.toFixed(0)}ms</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
