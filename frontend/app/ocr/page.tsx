"use client";

import { useState, useCallback } from "react";
import { Upload, FileText, Loader2, AlertCircle } from "lucide-react";
import { apiPost } from "@/lib/api";
import { OCRResponse } from "@/lib/types";

export default function OCRPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OCRResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      const data = await apiPost("/api/ocr", formData);
      setResult(data as OCRResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "OCR processing failed");
    } finally {
      setLoading(false);
    }
  }, [file]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">OCR — Text Extraction</h1>
        <p className="text-dark-200">Extract text from images using AI</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <div className="rounded-xl border border-dashed border-dark-500 bg-dark-800 p-8 text-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="ocr-upload"
            />
            <label htmlFor="ocr-upload" className="cursor-pointer">
              <FileText className="mx-auto h-12 w-12 text-dark-200" />
              <p className="mt-4 text-dark-50">Upload an image for OCR</p>
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

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="flex w-full items-center justify-center space-x-2 rounded-lg bg-primary-600 py-3 text-sm font-medium text-white hover:bg-primary-500 disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <FileText className="h-5 w-5" />
            )}
            <span>{loading ? "Extracting..." : "Extract Text"}</span>
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
            <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
              <h3 className="mb-4 font-semibold text-dark-50">Extracted Text</h3>
              <div className="rounded-lg bg-dark-900 p-4">
                <pre className="whitespace-pre-wrap text-sm text-dark-50">
                  {result.text || "No text detected"}
                </pre>
              </div>
              <div className="mt-4 flex items-center justify-between text-sm">
                <span className="text-dark-200">Confidence</span>
                <span className="text-dark-50">{(result.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="mt-2 flex items-center justify-between text-sm">
                <span className="text-dark-200">Processing Time</span>
                <span className="text-dark-50">{result.processing_time_ms.toFixed(0)}ms</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
