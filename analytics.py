"""Analytics dashboard data layer.

Provides a single, clean facade over the rolling detection history, system
monitoring and CSV/PDF export used by the GUI dashboard. The worker thread
pushes frame results here; the GUI reads aggregated statistics without
blocking the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils import (
    DetectionHistory,
    SystemMonitor,
    export_detections,
    export_pdf,
    timestamp,
)


@dataclass
class SessionAnalytics:
    """Aggregated analytics: history, system load, and export helpers.

    ``record`` is called from the worker thread (append-only). All read
    methods are safe to call from the GUI thread.
    """

    history: DetectionHistory = field(default_factory=DetectionHistory)
    monitor: SystemMonitor = field(default_factory=SystemMonitor)

    # -- ingestion (worker thread) ------------------------------------
    def record(self, frame_index: int, detections: list[Any]) -> None:
        self.history.record(frame_index, detections)

    # -- reads (GUI thread) -------------------------------------------
    def system_load(self) -> dict[str, float]:
        return self.monitor.snapshot()

    def label_counts(self) -> dict[str, int]:
        return self.history.label_counts()

    def source_counts(self) -> dict[str, int]:
        return self.history.source_counts()

    def confidence_stats(self) -> dict[str, float]:
        return self.history.confidence_stats()

    def session_stats(self) -> dict[str, Any]:
        return self.history.session_stats()

    def recent(self, limit: int = 60) -> list[dict[str, Any]]:
        return self.history.rows[-limit:]

    # -- exports ------------------------------------------------------
    def export_csv(self, directory: Path) -> Path:
        path = directory / f"detections-{timestamp()}.csv"
        return export_detections(self.history.rows, path)

    def export_pdf(self, directory: Path) -> Path:
        path = directory / f"report-{timestamp()}.pdf"
        return export_pdf(self.history, path)

    def reset(self) -> None:
        self.history.reset()


__all__ = ["SessionAnalytics"]
