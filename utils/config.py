# =============================================================================
# AI Vision Tracker Pro — Configuration Module
# =============================================================================
# Contains color palettes, COCO class definitions, category groups,
# and all rendering/UI constants used throughout the application.
# =============================================================================

import numpy as np

# ── Predefined colors for different classes (BGR format for OpenCV) ──────────
CLASS_COLORS = [
    (255, 56, 56),   # Red
    (255, 157, 151), # Light Red
    (255, 112, 31),  # Orange
    (255, 178, 29),  # Yellow
    (207, 210, 49),  # Olive
    (72, 249, 10),   # Green
    (146, 204, 23),  # Lime
    (61, 219, 134),  # Sea Green
    (26, 147, 52),   # Dark Green
    (0, 212, 187),   # Cyan
    (44, 153, 168),  # Teal
    (0, 194, 255),   # Light Blue
    (52, 69, 147),   # Dark Blue
    (100, 115, 255), # Purple-Blue
    (0, 24, 236),    # Deep Blue
    (132, 56, 255),  # Violet
    (82, 0, 133),    # Dark Violet
    (203, 56, 255),  # Magenta
    (255, 149, 200), # Pink
    (255, 55, 199)   # Hot Pink
]

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
    """Return a consistent color for a given class ID."""
    return CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]


def get_color_rgb(class_id):
    """Return color in RGB order (for Streamlit / matplotlib)."""
    b, g, r = get_color(class_id)
    return (r, g, b)


# ── Modern UI Settings for OpenCV Rendering ──────────────────────────────────
UI_SETTINGS = {
    "font": 2,               # cv2.FONT_HERSHEY_COMPLEX
    "font_scale": 0.55,
    "thickness": 2,
    "box_thickness": 2,
    "alpha": 0.55,            # Transparency for background panels
    "bg_color": (15, 15, 15),
    "hud_height": 70,         # Height of the top HUD bar in pixels
    "trail_max_len": 50,      # Maximum number of trail points to draw
}

# ── Alert configuration ─────────────────────────────────────────────────────
ALERT_CLASSES = ["person", "car", "truck", "knife", "scissors"]
ALERT_COOLDOWN_SEC = 5  # Minimum seconds between repeated alerts for same class
