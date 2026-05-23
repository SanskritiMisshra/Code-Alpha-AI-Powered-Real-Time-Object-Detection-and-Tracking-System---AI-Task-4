# =============================================================================
# AI Vision Tracker Pro — Video Stream Processor
# =============================================================================
# Manages OpenCV VideoCapture, FPS calculation, video recording,
# screenshot saving, and source metadata extraction.
# =============================================================================

import cv2
import time
import os
from collections import deque

OUTPUT_DIR = os.environ.get("TRACKER_OUTPUT_DIR", "output")


class VideoStreamProcessor:
    """Handles video input, frame reading, FPS calculation, recording, and screenshots."""

    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        # Try to set a reasonable resolution for webcams
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # FPS calculation using a rolling window
        self._fps_window = deque(maxlen=30)
        self._last_time = time.time()
        self.fps = 0.0

        # Source metadata
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame_idx = 0

        # Recording state
        self.writer = None
        self.recording = False
        self.output_path = None

    # ── Public API ───────────────────────────────────────────────────────────

    def is_opened(self):
        return self.cap.isOpened()

    def read_frame(self):
        """Read one frame; returns None when the source is exhausted."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.current_frame_idx += 1

        # Rolling-window FPS
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self._fps_window.append(1.0 / dt)
            self.fps = sum(self._fps_window) / len(self._fps_window)

        return frame

    def get_progress(self):
        """Returns (current_frame, total_frames) for progress bars."""
        if self.total_frames > 0:
            return self.current_frame_idx, self.total_frames
        return self.current_frame_idx, -1

    # ── Recording ────────────────────────────────────────────────────────────

    def start_recording(self, width=None, height=None, fps=None):
        """Begin writing processed frames to an MP4 file."""
        if self.recording:
            return self.output_path

        w = width or self.width
        h = height or self.height
        f = fps or min(self.source_fps, 30)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time())
        self.output_path = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, f, (w, h))
        self.recording = True
        return self.output_path

    def write_frame(self, frame):
        if self.recording and self.writer is not None:
            self.writer.write(frame)

    def stop_recording(self):
        if self.recording and self.writer is not None:
            self.writer.release()
            self.writer = None
            self.recording = False
            return self.output_path
        return None

    # ── Screenshots ──────────────────────────────────────────────────────────

    @staticmethod
    def save_screenshot(frame, prefix="screenshot"):
        """Save a single annotated frame as JPEG. Returns the file path."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(OUTPUT_DIR, f"{prefix}_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def release(self):
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
