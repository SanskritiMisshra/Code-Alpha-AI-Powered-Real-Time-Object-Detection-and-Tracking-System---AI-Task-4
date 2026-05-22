# =============================================================================
# AI Vision Tracker Pro — Tracker Manager
# =============================================================================
# Maintains per-object trail history, per-class counting (both cumulative and
# per-frame), and unique object registry across the entire session.
# =============================================================================

from collections import defaultdict
import numpy as np


class TrackerManager:
    """
    Manages tracking history, movement trails, and counting statistics.
    Works hand-in-hand with the YOLOv8 built-in ByteTrack tracker.
    """

    def __init__(self, max_trail_length=50):
        # trail history: track_id → deque-like list of (x, y) centers
        self.trails = defaultdict(list)
        self.max_trail_length = max_trail_length

        # Object registry
        self.unique_objects = set()                    # all track_ids ever seen
        self.unique_objects_by_class = defaultdict(set) # class_name → set of track_ids

        # Per-frame state (refreshed every update)
        self.current_class_counts = {}
        self.current_track_ids = set()

        # Cumulative class counts
        self.cumulative_class_counts = defaultdict(int)

        # Dwell time tracking (frames spent in scene)
        self.first_seen = {}   # track_id → frame_number
        self.last_seen = {}    # track_id → frame_number
        self._frame_num = 0

    # ── Core update ──────────────────────────────────────────────────────────

    def update(self, boxes, class_ids, track_ids, class_names):
        """
        Call once per frame with current detections.
        Updates trails, counts, and dwell-time tracking.
        """
        self._frame_num += 1
        self.current_class_counts = {}
        self.current_track_ids = set()

        for i, box in enumerate(boxes):
            tid = track_ids[i] if track_ids is not None and i < len(track_ids) else None
            if tid is None:
                continue

            tid = int(tid)
            cls_id = int(class_ids[i])
            cls_name = class_names[cls_id]

            # --- Registry ---
            is_new = tid not in self.unique_objects
            self.unique_objects.add(tid)
            self.unique_objects_by_class[cls_name].add(tid)
            self.current_track_ids.add(tid)

            if is_new:
                self.cumulative_class_counts[cls_name] += 1

            # --- Dwell time ---
            if tid not in self.first_seen:
                self.first_seen[tid] = self._frame_num
            self.last_seen[tid] = self._frame_num

            # --- Per-frame class counts ---
            self.current_class_counts[cls_name] = self.current_class_counts.get(cls_name, 0) + 1

            # --- Trail ---
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.trails[tid].append((cx, cy))
            if len(self.trails[tid]) > self.max_trail_length:
                self.trails[tid].pop(0)

    # ── Getters ──────────────────────────────────────────────────────────────

    def get_trails(self):
        return dict(self.trails)

    def get_total_unique_count(self):
        return len(self.unique_objects)

    def get_current_count(self):
        return len(self.current_track_ids)

    def get_current_class_counts(self):
        return dict(self.current_class_counts)

    def get_cumulative_class_counts(self):
        return dict(self.cumulative_class_counts)

    def get_unique_per_class(self):
        """Returns {class_name: number_of_unique_ids}."""
        return {k: len(v) for k, v in self.unique_objects_by_class.items()}

    def get_dwell_frames(self, track_id):
        """Returns number of frames a track_id has been visible."""
        if track_id in self.first_seen and track_id in self.last_seen:
            return self.last_seen[track_id] - self.first_seen[track_id] + 1
        return 0

    def get_avg_dwell(self):
        """Average dwell time across all tracked objects (in frames)."""
        if not self.first_seen:
            return 0
        dwells = [self.last_seen[tid] - self.first_seen[tid] + 1 for tid in self.first_seen]
        return np.mean(dwells)

    def reset(self):
        """Full reset — use when switching video sources."""
        self.trails.clear()
        self.unique_objects.clear()
        self.unique_objects_by_class.clear()
        self.current_class_counts.clear()
        self.current_track_ids.clear()
        self.cumulative_class_counts.clear()
        self.first_seen.clear()
        self.last_seen.clear()
        self._frame_num = 0
