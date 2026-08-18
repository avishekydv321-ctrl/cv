"""Lightweight centroid tracker for stable IDs across frames."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from utils import Detection


@dataclass
class Track:
    track_id: int
    label: str
    centroid: tuple[int, int]
    missed: int = 0


class CentroidTracker:
    """Greedy nearest-centroid association — fast and dependency free."""

    def __init__(self, max_missed: int = 12, max_distance: float = 90.0) -> None:
        self.max_missed = max_missed
        self.max_distance = max_distance
        self._next_id = 1
        self.tracks: "OrderedDict[int, Track]" = OrderedDict()

    @staticmethod
    def _centroid(det: Detection) -> tuple[int, int]:
        x1, y1, x2, y2 = det.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        centroids = [self._centroid(d) for d in detections]

        if not centroids:
            for track in list(self.tracks.values()):
                track.missed += 1
                if track.missed > self.max_missed:
                    del self.tracks[track.track_id]
            return dict(self.tracks)

        unmatched = set(range(len(centroids)))
        for track in self.tracks.values():
            if not unmatched:
                break
            best, best_dist = None, self.max_distance
            for index in unmatched:
                dist = float(np.hypot(*(np.array(centroids[index]) - np.array(track.centroid))))
                if dist < best_dist:
                    best, best_dist = index, dist
            if best is None:
                track.missed += 1
            else:
                track.centroid = centroids[best]
                track.label = detections[best].label
                track.missed = 0
                unmatched.discard(best)

        for index in unmatched:
            self.tracks[self._next_id] = Track(self._next_id, detections[index].label, centroids[index])
            self._next_id += 1

        for track_id, track in list(self.tracks.items()):
            if track.missed > self.max_missed:
                del self.tracks[track_id]

        return dict(self.tracks)

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
