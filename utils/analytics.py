# =============================================================================
# AI Vision Tracker Pro — Analytics Engine
# =============================================================================
# Provides zone intrusion detection, line-crossing counting,
# heatmap accumulation, speed estimation, and alert management.
# =============================================================================

import numpy as np
import cv2
import time
from collections import defaultdict
from .config import ALERT_CLASSES, ALERT_COOLDOWN_SEC


class ZoneDetector:
    """Checks whether object centers fall inside a user-defined polygon."""

    def __init__(self, zone_points):
        """zone_points: list of (x, y) tuples defining the polygon."""
        self.zone_points = np.array(zone_points, dtype=np.float32)
        self.intrusion_ids = set()

    def check(self, boxes, track_ids):
        """Returns set of track_ids currently inside the zone."""
        inside = set()
        for i, box in enumerate(boxes):
            tid = track_ids[i] if track_ids is not None and i < len(track_ids) else None
            if tid is None:
                continue
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            result = cv2.pointPolygonTest(self.zone_points, (cx, cy), False)
            if result >= 0:
                inside.add(int(tid))
        self.intrusion_ids = inside
        return inside

    @property
    def count(self):
        return len(self.intrusion_ids)


class LineCrossingCounter:
    """Counts objects crossing a horizontal or arbitrary line."""

    def __init__(self, line_start, line_end):
        self.line_start = np.array(line_start, dtype=np.float64)
        self.line_end = np.array(line_end, dtype=np.float64)
        self.count_in = 0
        self.count_out = 0
        self._prev_side = {}  # track_id -> side (+1 or -1)

    def _side(self, point):
        """Determine which side of the line a point is on."""
        d = (self.line_end - self.line_start)
        n = np.array([-d[1], d[0]])  # normal
        return np.sign(np.dot(np.array(point) - self.line_start, n))

    def update(self, boxes, track_ids):
        """Call every frame with current detections."""
        for i, box in enumerate(boxes):
            tid = track_ids[i] if track_ids is not None and i < len(track_ids) else None
            if tid is None:
                continue
            tid = int(tid)
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            side = self._side((cx, cy))

            if tid in self._prev_side:
                if self._prev_side[tid] != side and side != 0:
                    if side > 0:
                        self.count_in += 1
                    else:
                        self.count_out += 1
            self._prev_side[tid] = side


class HeatmapAccumulator:
    """Accumulates a density map of object positions over time."""

    def __init__(self, width, height, decay=0.995):
        self.width = width
        self.height = height
        self.decay = decay
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def update(self, boxes):
        """Add Gaussian blobs at each object center."""
        self.heatmap *= self.decay  # Slow decay so old positions fade
        for box in boxes:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                # Add a small Gaussian blob
                cv2.circle(self.heatmap, (cx, cy), 30, 1.0, -1)
        # Smooth
        self.heatmap = cv2.GaussianBlur(self.heatmap, (51, 51), 0)

    def get(self):
        return self.heatmap

    def reset(self):
        self.heatmap[:] = 0


class SpeedEstimator:
    """Estimates per-object speed in pixels/frame using trailing positions."""

    def __init__(self, smoothing_window=5):
        self.positions = defaultdict(list)
        self.window = smoothing_window
        self.speeds = {}

    def update(self, boxes, track_ids):
        for i, box in enumerate(boxes):
            tid = track_ids[i] if track_ids is not None and i < len(track_ids) else None
            if tid is None:
                continue
            tid = int(tid)
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            self.positions[tid].append((cx, cy))
            if len(self.positions[tid]) > self.window:
                self.positions[tid].pop(0)

            # Average displacement over window
            pts = self.positions[tid]
            if len(pts) >= 2:
                dists = [np.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1])
                         for j in range(1, len(pts))]
                self.speeds[tid] = np.mean(dists)
            else:
                self.speeds[tid] = 0.0

    def get_speeds(self):
        return dict(self.speeds)


class AlertManager:
    """Generates alerts when certain classes appear, with a cooldown."""

    def __init__(self, alert_classes=None, cooldown=ALERT_COOLDOWN_SEC):
        self.alert_classes = alert_classes or ALERT_CLASSES
        self.cooldown = cooldown
        self._last_alert = {}  # class_name -> timestamp
        self.active_alert = None

    def check(self, class_ids, class_names):
        """Returns alert string or None."""
        now = time.time()
        self.active_alert = None
        for cid in class_ids:
            name = class_names[int(cid)]
            if name in self.alert_classes:
                last = self._last_alert.get(name, 0)
                if now - last > self.cooldown:
                    self._last_alert[name] = now
                    self.active_alert = f"⚠ ALERT: {name.upper()} DETECTED"
                    return self.active_alert
        return self.active_alert


