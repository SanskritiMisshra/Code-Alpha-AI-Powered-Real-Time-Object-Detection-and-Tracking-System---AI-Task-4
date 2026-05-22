# =============================================================================
# AI Vision Tracker Pro — YOLOv8 Detection Engine
# =============================================================================
# Wraps Ultralytics YOLOv8 for single-frame detection + tracking.
# Supports model hot-swapping, GPU/CPU auto-selection, and custom IOU threshold.
# =============================================================================

from ultralytics import YOLO
import os
import torch
import numpy as np


class YOLOv8Detector:
    """High-level wrapper around Ultralytics YOLOv8 model."""

    MODEL_INFO = {
        "yolov8n.pt": {"desc": "Nano — Fastest, lowest accuracy",   "params": "3.2 M"},
        "yolov8s.pt": {"desc": "Small — Balanced speed & accuracy", "params": "11.2 M"},
        "yolov8m.pt": {"desc": "Medium — Higher accuracy, slower",  "params": "25.9 M"},
    }

    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLOv8 model.
        Automatically downloads the weights if they don't exist locally.
        Selects GPU if available, otherwise falls back to CPU.
        """
        os.makedirs("models", exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Loading {model_name} on {self.device.upper()}")

        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.model_name = model_name

    # ── Public API ───────────────────────────────────────────────────────────

    def get_classes(self):
        """Returns the dict of {class_id: class_name}."""
        return self.model.names

    def detect_only(self, frame, conf_threshold=0.25, iou_threshold=0.45, classes=None):
        """
        Run detection WITHOUT tracking (useful for single images / screenshots).
        Returns boxes, class_ids, confidences.
        """
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            verbose=False,
        )
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            return boxes, class_ids, confidences
        return np.array([]), np.array([]), np.array([])

    def detect_and_track(self, frame, conf_threshold=0.25, iou_threshold=0.45,
                         classes=None):
        """
        Runs YOLOv8 detection + built-in ByteTrack tracking on a frame.
        persist=True keeps IDs consistent across sequential calls.
        Returns (boxes, class_ids, track_ids, confidences).
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().numpy()
            else:
                track_ids = np.array([None] * len(boxes))
        else:
            boxes = np.array([])
            class_ids = np.array([])
            confidences = np.array([])
            track_ids = np.array([])

        return boxes, class_ids, track_ids, confidences

    def reset_tracker(self):
        """Reset the internal ByteTrack state (e.g. when switching video source)."""
        self.model.predictor = None
