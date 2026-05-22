# =============================================================================
# AI Vision Tracker Pro — Advanced Visualizer
# =============================================================================
# Futuristic OpenCV drawing functions: glowing boxes, gradient trails,
# HUD overlay, heatmap compositing, zone overlays, and line-crossing graphics.
# =============================================================================

import cv2
import numpy as np
import math
from .config import get_color, UI_SETTINGS


# ─── Glow effect helper ─────────────────────────────────────────────────────

def _add_glow(img, pt1, pt2, color, intensity=0.35, blur_size=21):
    """Draws a soft glow around a rectangle region."""
    glow = np.zeros_like(img, dtype=np.uint8)
    cv2.rectangle(glow, pt1, pt2, color, -1)
    glow = cv2.GaussianBlur(glow, (blur_size, blur_size), 0)
    cv2.addWeighted(glow, intensity, img, 1.0, 0, img)


# ─── Rounded-corner box with edge highlights ────────────────────────────────

def draw_modern_box(img, pt1, pt2, color, thickness, r=12, d=22):
    """Draw a bounding box with rounded corners and futuristic edge highlights."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Ensure the box is large enough for corner decorations
    w = x2 - x1
    h = y2 - y1
    r = min(r, w // 4, h // 4, 12)
    d = min(d, w // 3, h // 3, 22)
    if r < 2 or d < 2:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    # Faint inner fill
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    # Subtle glow behind the box
    _add_glow(img, pt1, pt2, color, intensity=0.12, blur_size=31)

    # Thin full rectangle
    cv2.rectangle(img, pt1, pt2, color, 1)

    # Corner accents — Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


# ─── Text with pill-shaped background ───────────────────────────────────────

def draw_label(img, text, position, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text on a pill-shaped translucent background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    pad = 6

    # Clamp so the label doesn't go off-screen
    y = max(th + pad * 2, y)
    x = max(0, x)

    # Pill background
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - th - pad), (x + tw + pad * 2, y + pad), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    cv2.putText(img, text, (x + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# ─── Gradient trail ─────────────────────────────────────────────────────────

def draw_gradient_trail(img, trail_points, color, max_thickness=6):
    """Draw a trail that fades from transparent to solid and thins over time."""
    n = len(trail_points)
    if n < 2:
        return
    for j in range(1, n):
        if trail_points[j - 1] is None or trail_points[j] is None:
            continue
        alpha = j / n  # 0→old, 1→new
        t = max(1, int(max_thickness * alpha))
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(img, trail_points[j - 1], trail_points[j], c, t, cv2.LINE_AA)


# ─── Speed badge ────────────────────────────────────────────────────────────

def draw_speed_badge(img, speed_px, position, color):
    """Draw a small speed indicator near the object."""
    if speed_px < 2:
        return
    text = f"{speed_px:.0f} px/f"
    draw_label(img, text, position, (255, 255, 255), color)


# ─── Main prediction renderer ───────────────────────────────────────────────

def render_predictions(frame, boxes, class_ids, track_ids, confidences,
                       class_names, show_trails=True, trails=None,
                       speeds=None, show_speed=False, face_blur=False):
    """
    Renders bounding boxes, labels, trails, speed badges on the frame.
    Optionally blurs faces (class 'person' eye region).
    """
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(class_ids[i])
        track_id = int(track_ids[i]) if track_ids is not None and track_ids[i] is not None else None
        conf = float(confidences[i])

        color = get_color(cls_id)

        # Face blur for persons
        if face_blur and class_names[cls_id] == "person":
            face_h = int((y2 - y1) * 0.3)
            face_region = frame[y1:y1 + face_h, x1:x2]
            if face_region.size > 0:
                face_region = cv2.GaussianBlur(face_region, (51, 51), 30)
                frame[y1:y1 + face_h, x1:x2] = face_region

        # Bounding box
        draw_modern_box(frame, (x1, y1), (x2, y2), color, UI_SETTINGS["box_thickness"])

        # Label
        cls_label = class_names[cls_id]
        if track_id is not None:
            label = f"#{track_id}  {cls_label}  {conf:.0%}"
        else:
            label = f"{cls_label}  {conf:.0%}"
        draw_label(frame, label, (x1, y1 - 4), (255, 255, 255), color)

        # Speed badge (bottom-right of box)
        if show_speed and speeds and track_id is not None and track_id in speeds:
            draw_speed_badge(frame, speeds[track_id], (x2 - 80, y2 + 18), color)

        # Gradient trail
        if show_trails and trails and track_id is not None and track_id in trails:
            draw_gradient_trail(frame, trails[track_id], color)

    return frame


# ─── Line-Crossing overlay ──────────────────────────────────────────────────

def draw_counting_line(frame, line_start, line_end, count_in, count_out):
    """Draw a counting line and its counters on the frame."""
    cv2.line(frame, line_start, line_end, (0, 255, 255), 2, cv2.LINE_AA)

    # Direction arrows
    mid_x = (line_start[0] + line_end[0]) // 2
    mid_y = (line_start[1] + line_end[1]) // 2
    draw_label(frame, f"IN: {count_in}", (mid_x - 60, mid_y - 20), (0, 255, 0), (0, 60, 0))
    draw_label(frame, f"OUT: {count_out}", (mid_x - 60, mid_y + 25), (0, 100, 255), (0, 30, 80))


# ─── Zone / Intrusion overlay ───────────────────────────────────────────────

def draw_zone(frame, zone_pts, intrusion_count=0, label="RESTRICTED"):
    """Draw a translucent polygon zone with intrusion counter."""
    pts = np.array(zone_pts, np.int32).reshape((-1, 1, 2))

    # Translucent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 0, 180))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    # Border
    cv2.polylines(frame, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)

    # Label
    cx = int(np.mean([p[0] for p in zone_pts]))
    cy = int(np.mean([p[1] for p in zone_pts]))
    draw_label(frame, f"{label} [{intrusion_count}]", (cx - 60, cy), (255, 255, 255), (0, 0, 180))

    if intrusion_count > 0:
        # Pulsing red border for active intrusion
        _add_glow(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), intensity=0.06, blur_size=61)


# ─── Heatmap compositing ────────────────────────────────────────────────────

def overlay_heatmap(frame, heatmap_accumulator, alpha=0.45):
    """Blend a heatmap onto the frame."""
    if heatmap_accumulator is None or heatmap_accumulator.max() == 0:
        return frame
    # Normalize to 0-255
    norm = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # Mask out zero regions so they stay transparent
    mask = (norm > 5).astype(np.uint8)
    mask_3c = cv2.merge([mask, mask, mask])
    blended = frame.copy()
    np.copyto(blended, cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0), where=mask_3c.astype(bool))
    return blended


# ─── HUD / Dashboard overlay ────────────────────────────────────────────────

def draw_dashboard(frame, fps, total_count, frame_count_dict, recording=False, alert_text=None):
    """Draws a modern sci-fi style HUD bar at the top of the frame."""
    h, w = frame.shape[:2]
    hud_h = UI_SETTINGS["hud_height"]

    # Semi-transparent dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), UI_SETTINGS["bg_color"], -1)
    cv2.addWeighted(overlay, UI_SETTINGS["alpha"], frame, 1 - UI_SETTINGS["alpha"], 0, frame)

    # Neon accent line under HUD
    cv2.line(frame, (0, hud_h), (w, hud_h), (0, 212, 187), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cyan = (187, 212, 0)  # BGR for cyan-ish
    white = (255, 255, 255)

    # FPS (with color coding)
    fps_color = (0, 255, 100) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 80, 255)
    cv2.putText(frame, f"FPS {fps:.1f}", (15, 28), font, 0.65, fps_color, 2, cv2.LINE_AA)

    # Total unique objects
    cv2.putText(frame, f"TRACKED  {total_count}", (15, 55), font, 0.55, cyan, 1, cv2.LINE_AA)

    # Per-class counts
    x_off = 200
    for cls_name, count in sorted(frame_count_dict.items(), key=lambda x: -x[1]):
        if count > 0 and x_off < w - 120:
            cv2.putText(frame, f"{cls_name}: {count}", (x_off, 28), font, 0.55, white, 1, cv2.LINE_AA)
            x_off += 130

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 30, 25), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Alert banner
    if alert_text:
        cv2.putText(frame, alert_text, (15, hud_h + 25), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        _add_glow(frame, (0, hud_h), (w, hud_h + 35), (0, 0, 255), 0.15, 21)

    return frame
