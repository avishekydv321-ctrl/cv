"use client";

import { useState } from "react";
import { Settings, Server, Key } from "lucide-react";
import { apiGet } from "@/lib/api";
import { HealthResponse, ModelsResponse } from "@/lib/types";

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [h, m] = await Promise.all([
          apiGet("/api/health"),
          apiGet("/api/models"),
        ]);
        setHealth(h);
        setModels(m);
      } catch (err) {
        console.error("Failed to load settings:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">Settings</h1>
        <p className="text-dark-200">System configuration and status</p>
      </div>

      {loading ? (
        <div className="flex h-64 items-center justify-center">
          <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary-400 border-t-transparent" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* System Info */}
          <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Server className="h-6 w-6 text-primary-400" />
              <h2 className="font-semibold text-dark-50">System Information</h2>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-dark-200">Backend Status</span>
                <span className="text-green-400">{health?.status}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-dark-200">Version</span>
                <span className="text-dark-50">{health?.version}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-dark-200">Device</span>
                <span className="text-dark-50">{health?.device}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-dark-200">Database</span>
                <span className={health?.database === "connected" ? "text-green-400" : "text-red-400"}>
                  {health?.database}
                </span>
              </div>
            </div>
          </div>

          {/* Model Status */}
          <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Key className="h-6 w-6 text-blue-400" />
              <h2 className="font-semibold text-dark-50">Model Status</h2>
            </div>
            <div className="max-h-64 overflow-y-auto space-y-2">
              {models?.models?.map((model) => (
                <div
                  key={model.key}
                  className="flex items-center justify-between rounded-lg bg-dark-700 px-3 py-2"
                >
                  <span className="text-sm text-dark-50">{model.name}</span>
                  <span
                    className={`text-xs ${
                      model.available ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {model.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Available Models */}
          <div className="rounded-xl border border-dark-500 bg-dark-800 p-6 lg:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <Settings className="h-6 w-6 text-yellow-400" />
              <h2 className="font-semibold text-dark-50">Configuration</h2>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <p className="text-sm text-dark-200">YOLO Weights Directory</p>
                <p className="font-mono text-sm text-dark-50">{models?.yolo_weights?.join(", ")}</p>
              </div>
              <div>
                <p className="text-sm text-dark-200">Inference Device</p>
                <p className="font-mono text-sm text-dark-50">{models?.device}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
