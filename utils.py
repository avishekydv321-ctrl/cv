"""Shared helpers: logging, FPS metering, filesystem paths, CSV/PDF export."""
from __future__ import annotations

import csv
import logging
import os
import shutil
import ssl
import subprocess
import tempfile
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, for both console and file sinks."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "visionsuite.log", encoding="utf-8"),
        ],
    )


def timestamp() -> str:
    """Filesystem-safe timestamp used for captures and recordings."""
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dirs() -> None:
    for directory in (OUTPUT_DIR, MODELS_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, timeout: int = 60) -> Optional[Path]:
    """Download a file to `destination` if it does not already exist.

    Returns the destination path on success, or None if the download failed
    (network error, timeout, etc.). Callers use this to pull optional model
    weights and simply report a feature as unavailable when it returns None.
    """
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("utils.download")
    temporary_path = None
    try:
        log.info("Downloading %s -> %s", url, destination)
        req = urllib.request.Request(url, headers={"User-Agent": "VisionSuite/1.0"})
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as out:
            temporary_path = Path(out.name)
        if shutil.which("curl"):
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--max-time", str(timeout), "--output", str(temporary_path), url],
                check=True,
                timeout=timeout + 5,
                capture_output=True,
                text=True,
            )
        else:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                with temporary_path.open("wb") as out:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        out.write(chunk)
        if temporary_path.stat().st_size == 0:
            raise OSError("downloaded file is empty")
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except Exception as exc:  # pragma: no cover - depends on network
        log.warning("Download failed for %s: %s", url, exc)
        return None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ensure_model(filename: str, url: str) -> Optional[Path]:
    """Return the path to a module weight file, downloading it if needed.

    If the file is absent and the download fails, returns None so the caller
    can degrade gracefully (mark the feature unavailable) instead of crashing.
    """
    path = MODELS_DIR / filename
    return download_file(url, path) if not path.exists() else path


@dataclass
class FPSMeter:
    """Rolling-window FPS estimator (cheap, allocation free)."""

    window: int = 30
    _stamps: Deque[float] = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self) -> None:
        self._stamps = deque(maxlen=self.window)

    def tick(self) -> float:
        self._stamps.append(time.perf_counter())
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    """One detected object/face/code in image coordinates."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    source: str = "object"
    color: Optional[tuple[int, int, int]] = None  # optional override for drawing

    def as_row(self, frame_index: int) -> dict[str, Any]:
        x1, y1, x2, y2 = self.box
        return {
            "frame": frame_index,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.source,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }


def export_detections(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    """Write detection rows to CSV. Returns the written path."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "time", "source", "label", "confidence", "x1", "y1", "x2", "y2"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def color_for(label: str) -> tuple[int, int, int]:
    """Deterministic BGR color per class label."""
    seed = abs(hash(label)) % (2**32)
    rng = np.random.default_rng(seed)
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


# ---------------------------------------------------------------------------
# System monitoring (CPU / RAM / GPU) — optional psutil, never crashes.
# ---------------------------------------------------------------------------
class SystemMonitor:
    """Lightweight snapshot of CPU, RAM and (optionally) GPU utilisation."""

    def __init__(self) -> None:
        self._psutil = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except Exception:
            self._psutil = None

    @property
    def available(self) -> bool:
        return self._psutil is not None

    def snapshot(self) -> dict[str, float]:
        """Return {'cpu', 'ram', 'gpu'} percentages (gpu=0.0 when unavailable)."""
        cpu = ram = gpu = 0.0
        if self._psutil is not None:
            cpu = float(self._psutil.cpu_percent(interval=None))
            ram = float(self._psutil.virtual_memory().percent)
        # GPU utilisation via torch (only when CUDA is active).
        try:
            import torch

            if torch.cuda.is_available():
                gpu = float(torch.cuda.utilization(0) if hasattr(torch.cuda, "utilization") else 0.0)
        except Exception:
            gpu = 0.0
        return {"cpu": cpu, "ram": ram, "gpu": gpu}


# ---------------------------------------------------------------------------
# Detection history + session analytics (rolling window, thread-safe-ish).
# ---------------------------------------------------------------------------
@dataclass
class DetectionHistory:
    """Rolling record of detections plus derived session statistics.

    ``record`` is called from the worker thread for each frame that produced
    detections; ``snapshot``/``stats`` are read from the GUI thread. The
    worker only ever appends and the GUI only ever reads, so no lock is needed
    for the common path (append is atomic under the GIL).
    """

    limit: int = 200
    rows: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record(self, frame_index: int, detections: Iterable[Detection]) -> None:
        for det in detections:
            self.rows.append(det.as_row(frame_index))
        if len(self.rows) > self.limit:
            self.rows = self.rows[-self.limit:]

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        return counts

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["source"]] = counts.get(row["source"], 0) + 1
        return counts

    def confidence_stats(self) -> dict[str, float]:
        confs = [float(r["confidence"]) for r in self.rows]
        if not confs:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
        return {
            "min": min(confs),
            "max": max(confs),
            "mean": float(np.mean(confs)),
            "count": len(confs),
        }

    def session_stats(self) -> dict[str, Any]:
        elapsed = max(time.time() - self.start_time, 0.0)
        return {
            "duration_s": round(elapsed, 1),
            "total_detections": len(self.rows),
            "unique_labels": len(self.label_counts()),
        }

    def reset(self) -> None:
        self.rows.clear()
        self.start_time = time.time()


# ---------------------------------------------------------------------------
# PDF report export (matplotlib-backed, large-font, dark themed).
# ---------------------------------------------------------------------------
def export_pdf(
    history: DetectionHistory,
    path: Path,
    title: str = "VisionSuite — Session Report",
) -> Path:
    """Render a multi-panel PDF report of the session analytics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 12), facecolor="#0e1117")
        fig.suptitle(title, color="#e6edf3", fontsize=18, fontweight="bold")

        # Panel 1: label histogram.
        ax1 = fig.add_subplot(2, 1, 1)
        counts = history.label_counts()
        if counts:
            labels = list(counts.keys())
            values = list(counts.values())
            ax1.bar(labels, values, color="#12a67c")
            ax1.set_title("Detections by label", color="#e6edf3")
        else:
            ax1.text(0.5, 0.5, "No detections recorded", ha="center", color="#e6edf3")
        ax1.tick_params(colors="#9fb0c3")
        ax1.set_facecolor("#131922")

        # Panel 2: confidence distribution.
        ax2 = fig.add_subplot(2, 1, 2)
        confs = [float(r["confidence"]) for r in history.rows]
        if confs:
            ax2.hist(confs, bins=20, color="#3a94ff")
            ax2.set_title("Confidence distribution", color="#e6edf3")
        else:
            ax2.text(0.5, 0.5, "No confidence data", ha="center", color="#e6edf3")
        ax2.tick_params(colors="#9fb0c3")
        ax2.set_facecolor("#131922")

        # Footer summary.
        stats = history.session_stats()
        fig.text(
            0.5, 0.02,
            f"Duration: {stats['duration_s']}s  |  "
            f"Detections: {stats['total_detections']}  |  "
            f"Labels: {stats['unique_labels']}",
            ha="center", color="#7ee7c7", fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(path, format="pdf", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - matplotlib may be missing
        log = logging.getLogger("utils.export")
        log.warning("PDF export failed: %s", exc)
        return path
