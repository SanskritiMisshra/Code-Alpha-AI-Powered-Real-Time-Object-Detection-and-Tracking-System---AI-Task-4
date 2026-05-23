# =============================================================================
# AI Vision Tracker Pro — Configuration Module
# =============================================================================
# Contains color palettes, COCO class definitions, category groups,
# and all rendering/UI constants used throughout the application.
# =============================================================================

import numpy as np

# ── Strict SOC Colors (BGR format for OpenCV) ────────────────────────────────
SOC_COLORS = {
    "person": (246, 130, 59),     # #3B82F6 (Blue)
    "vehicle": (129, 185, 16),    # #10B981 (Green)
    "alert": (68, 68, 239),       # #EF4444 (Red)
    "unknown": (11, 158, 245)     # #F59E0B (Orange)
}

# ── COCO 80 Dataset Classes ─────────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ── Semantic class groups for quick filtering ────────────────────────────────
CLASS_GROUPS = {
    "People":   ["person"],
    "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "Animals":  ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "Electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "Furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "Food":     ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "Kitchen":  ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "microwave", "oven", "toaster", "sink", "refrigerator"],
}

def get_group_class_ids(group_name):
    """Return COCO class IDs for a named group."""
    names = CLASS_GROUPS.get(group_name, [])
    return [COCO_CLASSES.index(n) for n in names if n in COCO_CLASSES]


def get_color(class_id):
    """Return a consistent SOC color based on class category."""
    try:
        name = COCO_CLASSES[int(class_id)]
        if name in ALERT_CLASSES:
            return SOC_COLORS["alert"]
        if name in CLASS_GROUPS.get("People", []):
            return SOC_COLORS["person"]
        if name in CLASS_GROUPS.get("Vehicles", []):
            return SOC_COLORS["vehicle"]
        return SOC_COLORS["unknown"]
    except:
        return SOC_COLORS["unknown"]


def get_color_rgb(class_id):
    """Return color in RGB order (for Streamlit / matplotlib)."""
    b, g, r = get_color(class_id)
    return (r, g, b)


# ── Modern UI Settings for OpenCV Rendering ──────────────────────────────────
UI_SETTINGS = {
    "font": 0,                # cv2.FONT_HERSHEY_SIMPLEX (Clean, modern)
    "font_scale": 0.50,
    "thickness": 1,
    "box_thickness": 1,       # Thin subtle borders
    "alpha": 0.80,            # Darker overlays
    "bg_color": (20, 15, 11), # #0B0F14 in BGR
    "hud_height": 40,         # Slimmer top HUD
    "trail_max_len": 40,
}

# ── Alert configuration ─────────────────────────────────────────────────────
ALERT_CLASSES = ["person", "car", "truck", "knife", "scissors"]
ALERT_COOLDOWN_SEC = 5  # Minimum seconds between repeated alerts for same class
