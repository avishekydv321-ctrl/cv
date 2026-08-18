"use client";

import { useEffect, useState } from "react";
import { BarChart3, TrendingUp } from "lucide-react";
import { apiGet } from "@/lib/api";
import { DashboardStats } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const COLORS = ["#00e6ac", "#3a94ff", "#ff6b6b", "#ffd93d", "#6bcb77"];

export default function AnalyticsPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await apiGet("/api/history/dashboard");
        setStats(data);
      } catch (err) {
        console.error("Failed to load analytics:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary-400 border-t-transparent" />
      </div>
    );
  }

  const chartData = stats?.most_detected_vehicles?.map((v) => ({
    name: v.type,
    value: v.count,
  })) || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-dark-50">Analytics</h1>
        <p className="text-dark-200">Detection statistics and insights</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Summary Cards */}
        <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
          <div className="flex items-center space-x-3">
            <TrendingUp className="h-8 w-8 text-primary-400" />
            <div>
              <p className="text-sm text-dark-200">Total Detections</p>
              <p className="text-3xl font-bold text-dark-50">{stats?.total_detections ?? 0}</p>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
          <div className="flex items-center space-x-3">
            <BarChart3 className="h-8 w-8 text-blue-400" />
            <div>
              <p className="text-sm text-dark-200">Today's Detections</p>
              <p className="text-3xl font-bold text-dark-50">{stats?.todays_detections ?? 0}</p>
            </div>
          </div>
        </div>

        {/* Vehicle Detection Chart */}
        <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
          <h3 className="mb-4 font-semibold text-dark-50">Vehicle Types</h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232b36" />
                <XAxis dataKey="name" stroke="#9fb0c3" />
                <YAxis stroke="#9fb0c3" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#131922",
                    border: "1px solid #232b36",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="value" fill="#00e6ac" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-center text-dark-200">No data available</p>
          )}
        </div>

        {/* Session Stats */}
        <div className="rounded-xl border border-dark-500 bg-dark-800 p-6">
          <h3 className="mb-4 font-semibold text-dark-50">Session Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-dark-200">Total Sessions</span>
              <span className="font-semibold text-dark-50">{stats?.total_sessions ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-dark-200">Unique Plates</span>
              <span className="font-semibold text-dark-50">{stats?.unique_plates ?? 0}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-dark-200">Total Faces</span>
              <span className="font-semibold text-dark-50">{stats?.total_faces ?? 0}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
