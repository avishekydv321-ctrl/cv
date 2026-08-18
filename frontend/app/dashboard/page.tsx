"use client";

import { useEffect, useState } from "react";
import { Activity, Camera, Image, Video, Car, FileText, Cpu } from "lucide-react";
import { apiGet } from "@/lib/api";
import { DashboardStats, HealthResponse } from "@/lib/types";

const statCards = [
  { title: "Total Detections", key: "total_detections", icon: Activity, color: "text-primary-400" },
  { title: "Today's Detections", key: "todays_detections", icon: Activity, color: "text-blue-400" },
  { title: "Vehicles Detected", key: "total_vehicles", icon: Car, color: "text-green-400" },
  { title: "Unique Plates", key: "unique_plates", icon: Car, color: "text-yellow-400" },
  { title: "Total Sessions", key: "total_sessions", icon: Camera, color: "text-purple-400" },
];

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [statsData, healthData] = await Promise.all([
          apiGet("/api/history/dashboard"),
          apiGet("/api/health"),
        ]);
        setStats(statsData);
        setHealth(healthData);
      } catch (error) {
        console.error("Failed to load dashboard:", error);
      } finally {
        setLoading(false);
      }
    }
    loadData()
  }, []);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary-400 border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">Dashboard</h1>
        <p className="text-dark-200">AI System Overview</p>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
          <div className="flex items-center space-x-3">
            <Cpu className="h-8 w-8 text-primary-400" />
            <div>
              <p className="text-sm text-dark-200">Device</p>
              <p className="text-lg font-semibold text-dark-50">{health?.device || "N/A"}</p>
            </div>
          </div>
        </div>
        <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
          <div className="flex items-center space-x-3">
            <Activity className="h-8 w-8 text-green-400" />
            <div>
              <p className="text-sm text-dark-200">Database</p>
              <p className={`text-lg font-semibold ${health?.database === "connected" ? "text-green-400" : "text-red-400"}`}>
                {health?.database === "connected" ? "Connected" : "Disconnected"}
              </p>
            </div>
          </div>
        </div>
        <div className="rounded-xl border border-dark-500 bg-dark-800 p-4">
          <div className="flex items-center space-x-3">
            <Camera className="h-8 w-8 text-blue-400" />
            <div>
              <p className="text-sm text-dark-200">Status</p>
              <p className="text-lg font-semibold text-green-400">Online</p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {statCards.map((card) => (
          <div
            key={card.key}
            className="rounded-xl border border-dark-500 bg-dark-800 p-5"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-dark-200">{card.title}</p>
                <p className="text-2xl font-bold text-dark-50">
                  {stats?.[card.key as keyof DashboardStats] ?? 0}
                </p>
              </div>
              <card.icon className={`h-8 w-8 ${card.color}`} />
            </div>
          </div>
        ))}
      </div>

      {/* Recent Detections */}
      <div className="rounded-xl border border-dark-500 bg-dark-800">
        <div className="border-b border-dark-500 px-6 py-4">
          <h2 className="text-lg font-semibold text-dark-50">Recent Detections</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-dark-500 text-left text-sm text-dark-200">
                <th className="px-6 py-3">Class</th>
                <th className="px-6 py-3">Confidence</th>
                <th className="px-6 py-3">Source</th>
                <th className="px-6 py-3">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-500">
              {stats?.recent_detections?.slice(0, 10).map((det, idx) => (
                <tr key={idx} className="text-sm">
                  <td className="px-6 py-3 font-medium text-dark-50">{det.object_class}</td>
                  <td className="px-6 py-3 text-dark-200">{(det.confidence * 100).toFixed(1)}%</td>
                  <td className="px-6 py-3 text-dark-200">{det.source || "—"}</td>
                  <td className="px-6 py-3 text-dark-200">
                    {det.detected_at ? new Date(det.detected_at).toLocaleString() : "—"}
                  </td>
                </tr>
              ))}
              {(!stats?.recent_detections || stats.recent_detections.length === 0) && (
                <tr>
                  <td colSpan={4} className="px-6 py-8 text-center text-dark-200">
                    No detections yet. Start the camera or upload an image to begin.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
