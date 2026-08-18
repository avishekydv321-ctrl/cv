"use client";

import { useEffect, useState } from "react";
import { History } from "lucide-react";
import { apiGet } from "@/lib/api";
import { HistoryResponse } from "@/lib/types";

export default function HistoryPage() {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);

  useEffect(() => {
    async function load() {
      try {
        const res = await apiGet(`/api/history/detections?page=${page}&page_size=20`);
        setData(res);
      } catch (err) {
        console.error("Failed to load history:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [page]);

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
        <h1 className="text-2xl font-bold text-dark-50">Detection History</h1>
        <p className="text-dark-200">View past detection records</p>
      </div>

      <div className="rounded-xl border border-dark-500 bg-dark-800">
        <div className="border-b border-dark-500 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <History className="h-5 w-5 text-primary-400" />
              <h2 className="font-semibold text-dark-50">Detection Records</h2>
            </div>
            <span className="text-sm text-dark-200">
              Total: {data?.total ?? 0}
            </span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-dark-500 text-left text-sm text-dark-200">
                <th className="px-6 py-3">Type</th>
                <th className="px-6 py-3">Class</th>
                <th className="px-6 py-3">Confidence</th>
                <th className="px-6 py-3">Source</th>
                <th className="px-6 py-3">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-500">
              {data?.items?.map((item) => (
                <tr key={item.id} className="text-sm">
                  <td className="px-6 py-3 text-dark-200">{item.detection_type}</td>
                  <td className="px-6 py-3 font-medium text-dark-50">{item.object_class}</td>
                  <td className="px-6 py-3 text-dark-200">{(item.confidence * 100).toFixed(1)}%</td>
                  <td className="px-6 py-3 text-dark-200">{item.source || "—"}</td>
                  <td className="px-6 py-3 text-dark-200">
                    {item.detected_at ? new Date(item.detected_at).toLocaleString() : "—"}
                  </td>
                </tr>
              ))}
              {(!data?.items || data.items.length === 0) && (
                <tr>
                  <td colSpan={5} className="px-6 py-8 text-center text-dark-200">
                    No records found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between border-t border-dark-500 px-6 py-4">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="rounded-lg border border-dark-500 px-4 py-2 text-sm text-dark-200 hover:bg-dark-700 disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-sm text-dark-200">Page {page}</span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={!data || data.items.length < data.page_size}
            className="rounded-lg border border-dark-500 px-4 py-2 text-sm text-dark-200 hover:bg-dark-700 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
