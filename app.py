# =============================================================================
# AI Vision Tracker Pro — Streamlit Application
# =============================================================================
# Production-grade dashboard with real-time video feed, advanced analytics,
# zone intrusion detection, line-crossing counting, heatmap overlay,
# speed estimation, face blur, recording, screenshots, and live charts.
# =============================================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import tempfile

from detector import YOLOv8Detector
from tracker import TrackerManager
from video_processor import VideoStreamProcessor
from utils.visualizer import (
    render_predictions, draw_dashboard, draw_counting_line,
    draw_zone, overlay_heatmap,
)
from utils.analytics import (
    ZoneDetector, LineCrossingCounter, HeatmapAccumulator,
    SpeedEstimator, AlertManager,
)
from utils.config import COCO_CLASSES, CLASS_GROUPS, get_group_class_ids

# ═════════════════════════════════════════════════════════════════════════════
# Page Config
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Vision Tracker Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# Custom CSS — Futuristic dark theme
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Global ────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --accent: #00d4bb;
        --accent-glow: rgba(0, 212, 187, 0.35);
        --danger: #ff4757;
        --bg-dark: #0a0e17;
        --bg-card: #111827;
        --bg-card-hover: #1a2332;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border-subtle: rgba(255,255,255,0.06);
    }

    .stApp {
        background: var(--bg-dark);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1623 0%, #0a0e17 100%);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--accent) !important;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* ── Headers ───────────────────────────────────────────── */
    h1 {
        background: linear-gradient(135deg, #00d4bb 0%, #00b4d8 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h2, h3, h4 {
        color: #e2e8f0 !important;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4bb 0%, #00b4d8 100%);
        color: #000 !important;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        box-shadow: 0 0 20px var(--accent-glow);
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 30px var(--accent-glow), 0 4px 15px rgba(0,0,0,0.3);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Metrics ───────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* ── Sliders / Inputs ──────────────────────────────────── */
    .stSlider > div > div > div > div {
        background: var(--accent) !important;
    }
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
    }

    /* ── Expander ──────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 8px;
    }

    /* ── Tabs ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card);
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        border: 1px solid var(--border-subtle);
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card-hover) !important;
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent);
    }

    /* ── Divider ───────────────────────────────────────────── */
    hr {
        border-color: var(--border-subtle) !important;
    }

    /* ── Toast / success / error ───────────────────────────── */
    .stAlert {
        border-radius: 8px;
    }

    /* ── Hide default hamburger & footer ───────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Custom stat card ──────────────────────────────────── */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .stat-card h4 {
        margin: 0;
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stat-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Header
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1rem;">
    <h1 style="font-size:2.6rem; margin-bottom:0;">👁️ AI Vision Tracker Pro</h1>
    <p style="color:#94a3b8; font-size:1rem; margin-top:0.2rem;">
        Real-Time Object Detection · ByteTrack Tracking · Advanced Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# Sidebar — Configuration
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── Model ────────────────────────────────────────────────
    st.markdown("#### 🧠 Model")
    model_choice = st.selectbox(
        "YOLOv8 Model",
        list(YOLOv8Detector.MODEL_INFO.keys()),
        index=0,
        format_func=lambda x: f"{x.replace('.pt','')}  —  {YOLOv8Detector.MODEL_INFO[x]['desc']}",
    )

    # ── Source ───────────────────────────────────────────────
    st.markdown("#### 📹 Source")
    source_type = st.radio("Input", ["Webcam", "Video File"], horizontal=True)
    video_file = None
    if source_type == "Video File":
        video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "webm"])

    # ── Detection ────────────────────────────────────────────
    st.markdown("#### 🎯 Detection")
    conf_thresh = st.slider("Confidence", 0.10, 1.0, 0.30, 0.05)
    iou_thresh  = st.slider("IOU Threshold", 0.10, 1.0, 0.45, 0.05)

    # Class filtering
    filter_mode = st.radio("Class Filter", ["All Classes", "By Group", "Custom"], horizontal=True)
    selected_class_ids = None
    if filter_mode == "By Group":
        group = st.selectbox("Group", list(CLASS_GROUPS.keys()))
        selected_class_ids = get_group_class_ids(group)
    elif filter_mode == "Custom":
        chosen = st.multiselect("Pick classes", COCO_CLASSES, default=["person", "car"])
        selected_class_ids = [COCO_CLASSES.index(n) for n in chosen]

    # ── Visualization ────────────────────────────────────────
    st.markdown("#### 🎨 Visualization")
    show_trails  = st.checkbox("Movement Trails", value=True)
    show_speed   = st.checkbox("Speed Badges", value=False)
    show_heatmap = st.checkbox("Heatmap Overlay", value=False)
    face_blur    = st.checkbox("Face Blur (Privacy)", value=False)

    # ── Advanced Features ────────────────────────────────────
    st.markdown("#### 🔬 Advanced")
    enable_line_counter = st.checkbox("Line-Crossing Counter", value=False)
    if enable_line_counter:
        line_y_pct = st.slider("Line Y position (%)", 10, 90, 50, 5,
                               help="Horizontal line position as % of frame height")
    else:
        line_y_pct = 50

    enable_zone = st.checkbox("Zone Intrusion Detection", value=False)
    enable_alerts = st.checkbox("Smart Alerts", value=False)

    # ── Controls ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎬 Controls")
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶️ Start")
    stop_btn  = c2.button("⏹️ Stop")
    record_toggle = st.checkbox("⏺️ Record Output", value=False)
    screenshot_btn = st.button("📸 Screenshot")


# ═════════════════════════════════════════════════════════════════════════════
# Session state
# ═════════════════════════════════════════════════════════════════════════════
if "running" not in st.session_state:
    st.session_state.running = False
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


# ═════════════════════════════════════════════════════════════════════════════
# Model loading (cached)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_detector(model_name):
    return YOLOv8Detector(model_name)

detector = load_detector(model_choice)
class_names = detector.get_classes()


# ═════════════════════════════════════════════════════════════════════════════
# Main layout — metrics row + video feed + analytics tabs
# ═════════════════════════════════════════════════════════════════════════════

# Top metric cards
m1, m2, m3, m4, m5 = st.columns(5)
fps_ph      = m1.empty()
current_ph  = m2.empty()
total_ph    = m3.empty()
line_ph     = m4.empty()
zone_ph     = m5.empty()

fps_ph.metric("FPS", "—")
current_ph.metric("On Screen", "—")
total_ph.metric("Total Tracked", "—")
line_ph.metric("Line In / Out", "— / —")
zone_ph.metric("Zone Intrusions", "—")

# Video feed
frame_placeholder = st.empty()

# Progress bar for video files
progress_placeholder = st.empty()

# Analytics section below video
st.markdown("---")
tab_stats, tab_classes, tab_info = st.tabs(["📊 Live Statistics", "🏷️ Class Breakdown", "ℹ️ System Info"])

with tab_stats:
    stats_placeholder = st.empty()

with tab_classes:
    class_chart_placeholder = st.empty()

with tab_info:
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | **Model** | `{model_choice}` ({YOLOv8Detector.MODEL_INFO[model_choice]['params']} parameters) |
    | **Tracker** | ByteTrack (built-in) |
    | **Device** | `{detector.device.upper()}` |
    | **Confidence** | {conf_thresh} |
    | **IOU** | {iou_thresh} |
    """)


# ═════════════════════════════════════════════════════════════════════════════
# Processing Loop
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.running:
    # ── Resolve source ───────────────────────────────────────
    if source_type == "Webcam":
        source = 0
    else:
        if video_file is None:
            st.error("⚠️ Please upload a video file first.")
            st.session_state.running = False
            st.stop()
        # Write uploaded bytes to a temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        tmp.close()  # Close the file so OpenCV can read it (important on Windows!)
        source = tmp.name

    # ── Initialize components ────────────────────────────────
    processor = VideoStreamProcessor(source)
    if not processor.is_opened():
        st.error(f"❌ Failed to open video source: {source}")
        st.session_state.running = False
        st.stop()

    tracker       = TrackerManager(max_trail_length=50)
    speed_est     = SpeedEstimator()
    alert_mgr     = AlertManager() if enable_alerts else None
    heatmap_acc   = HeatmapAccumulator(processor.width, processor.height) if show_heatmap else None

    # Line crossing
    line_counter = None
    if enable_line_counter:
        ly = int(processor.height * line_y_pct / 100)
        line_counter = LineCrossingCounter((0, ly), (processor.width, ly))

    # Zone intrusion (default: a rectangle in the center)
    zone_det = None
    if enable_zone:
        w, h = processor.width, processor.height
        zone_pts = [
            (int(w * 0.3), int(h * 0.3)),
            (int(w * 0.7), int(h * 0.3)),
            (int(w * 0.7), int(h * 0.7)),
            (int(w * 0.3), int(h * 0.7)),
        ]
        zone_det = ZoneDetector(zone_pts)

    # Recording
    if record_toggle:
        out_path = processor.start_recording()
        st.toast(f"Recording started → {out_path}", icon="⏺️")

    # ── Frame loop ───────────────────────────────────────────
    last_annotated = None

    while st.session_state.running:
        frame = processor.read_frame()
        if frame is None:
            st.toast("🏁 Video stream ended.", icon="✅")
            break

        # Detect + Track
        boxes, class_ids, track_ids, confs = detector.detect_and_track(
            frame,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            classes=selected_class_ids,
        )

        # Update tracker
        tracker.update(boxes, class_ids, track_ids, class_names)

        # Speed estimation
        speed_est.update(boxes, track_ids)

        # Heatmap
        if heatmap_acc is not None:
            heatmap_acc.update(boxes)

        # Line crossing
        if line_counter is not None:
            line_counter.update(boxes, track_ids)

        # Zone intrusion
        zone_count = 0
        if zone_det is not None:
            zone_det.check(boxes, track_ids)
            zone_count = zone_det.count

        # Alert
        alert_text = None
        if alert_mgr is not None:
            alert_text = alert_mgr.check(class_ids, class_names)

        # ── Render ───────────────────────────────────────────
        vis = frame.copy()

        # Heatmap first (below everything)
        if heatmap_acc is not None:
            vis = overlay_heatmap(vis, heatmap_acc.get(), alpha=0.35)

        # Zone
        if zone_det is not None:
            draw_zone(vis, zone_pts, zone_count)

        # Line
        if line_counter is not None:
            draw_counting_line(
                vis,
                (0, int(processor.height * line_y_pct / 100)),
                (processor.width, int(processor.height * line_y_pct / 100)),
                line_counter.count_in,
                line_counter.count_out,
            )

        # Predictions
        vis = render_predictions(
            vis, boxes, class_ids, track_ids, confs, class_names,
            show_trails=show_trails,
            trails=tracker.get_trails(),
            speeds=speed_est.get_speeds() if show_speed else None,
            show_speed=show_speed,
            face_blur=face_blur,
        )

        # HUD
        vis = draw_dashboard(
            vis, processor.fps,
            tracker.get_total_unique_count(),
            tracker.get_current_class_counts(),
            recording=processor.recording,
            alert_text=alert_text,
        )

        last_annotated = vis.copy()

        # Write recording frame
        processor.write_frame(vis)

        # ── Display ──────────────────────────────────────────
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB")

        # Metrics
        fps_ph.metric("FPS", f"{processor.fps:.1f}")
        current_ph.metric("On Screen", tracker.get_current_count())
        total_ph.metric("Total Tracked", tracker.get_total_unique_count())

        if line_counter is not None:
            line_ph.metric("Line In / Out", f"{line_counter.count_in} / {line_counter.count_out}")
        zone_ph.metric("Zone Intrusions", zone_count)

        # Progress (video files only)
        cur, tot = processor.get_progress()
        if tot > 0:
            progress_placeholder.progress(min(cur / tot, 1.0), text=f"Frame {cur}/{tot}")

        # Live stats table
        unique_per_class = tracker.get_unique_per_class()
        if unique_per_class:
            stats_placeholder.dataframe(
                {
                    "Class": list(unique_per_class.keys()),
                    "Unique IDs": list(unique_per_class.values()),
                    "Currently Visible": [tracker.get_current_class_counts().get(k, 0) for k in unique_per_class],
                },

                hide_index=True,
            )

        # Class breakdown chart
        cc = tracker.get_current_class_counts()
        if cc:
            import pandas as pd
            df = pd.DataFrame({"Class": list(cc.keys()), "Count": list(cc.values())})
            class_chart_placeholder.bar_chart(df.set_index("Class"))

        # Screenshot
        if screenshot_btn and last_annotated is not None:
            ss_path = VideoStreamProcessor.save_screenshot(last_annotated)
            st.toast(f"📸 Saved: {ss_path}", icon="✅")

        time.sleep(0.005)  # Yield to Streamlit event loop

    # ── Cleanup ──────────────────────────────────────────────
    if record_toggle:
        saved = processor.stop_recording()
        if saved:
            st.toast(f"💾 Recording saved: {saved}", icon="🎬")
    processor.release()
    st.session_state.running = False

elif not st.session_state.running:
    # Show a welcome / idle state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; 
                background: linear-gradient(135deg, #111827 0%, #0a0e17 100%);
                border: 1px solid rgba(255,255,255,0.06); border-radius: 16px;
                margin: 1rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
        <h2 style="color: #00d4bb; margin-bottom: 0.5rem;">Ready to Detect</h2>
        <p style="color: #94a3b8; font-size: 1.05rem; max-width: 500px; margin: 0 auto;">
            Select your video source and configure detection parameters in the sidebar, 
            then press <strong style="color:#00d4bb;">▶️ Start</strong> to begin real-time tracking.
        </p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div class="stat-card" style="min-width: 140px;">
                <h4>Model</h4>
                <div class="value" style="font-size:1rem;">""" + model_choice.replace('.pt','').upper() + """</div>
            </div>
            <div class="stat-card" style="min-width: 140px;">
                <h4>Device</h4>
                <div class="value" style="font-size:1rem;">""" + detector.device.upper() + """</div>
            </div>
            <div class="stat-card" style="min-width: 140px;">
                <h4>Classes</h4>
                <div class="value" style="font-size:1rem;">80 COCO</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
