# 👁️ AI Vision Tracker Pro

A **production-grade**, real-time object detection and tracking system built with **YOLOv8**, **ByteTrack**, **OpenCV**, and **Streamlit**. Features a futuristic dark-mode dashboard with advanced computer-vision analytics.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)

---

## ✨ Features

### Core
| Feature | Description |
|---------|-------------|
| **Real-Time Detection** | YOLOv8 (Nano / Small / Medium) with GPU auto-detection |
| **Object Tracking** | ByteTrack assigns unique IDs that persist across frames |
| **Movement Trails** | Gradient-colored trails visualize each object's path |
| **Speed Estimation** | Per-object speed badges (pixels/frame) |
| **Face Blur** | One-click privacy mode blurs detected faces |

### Advanced Analytics
| Feature | Description |
|---------|-------------|
| **Line-Crossing Counter** | Counts objects crossing a configurable horizontal line (IN / OUT) |
| **Zone Intrusion Detection** | Polygon-based restricted area monitoring with alert glow |
| **Heatmap Overlay** | Accumulated density map showing where objects spend time |
| **Smart Alerts** | Triggers on-screen warnings when specific classes appear |
| **Live Charts** | Real-time bar charts and per-class breakdown tables |

### Interface
| Feature | Description |
|---------|-------------|
| **Modern Dashboard** | Futuristic dark theme with glowing accents and neon HUD |
| **Class Filtering** | Filter by group (People, Vehicles, Animals…) or pick individual classes |
| **Recording** | Save annotated video output as MP4 |
| **Screenshots** | Capture the current annotated frame as JPEG |
| **Progress Bar** | Video file playback progress indicator |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the App

```bash
streamlit run app.py
```

### 3. Start Detecting

1. Select a **video source** (Webcam or upload a file) in the sidebar.
2. Choose the **YOLOv8 model size** (nano = fastest, medium = most accurate).
3. Enable desired features (trails, heatmap, line counter, etc.).
4. Click **▶️ Start** and watch the magic happen.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web application
├── detector.py             # YOLOv8 detection engine
├── tracker.py              # Tracking manager (trails, counts, dwell time)
├── video_processor.py      # Video capture, FPS, recording, screenshots
├── utils/
│   ├── __init__.py
│   ├── config.py           # Colors, class groups, UI constants
│   ├── visualizer.py       # OpenCV rendering (glow boxes, HUD, heatmap)
│   └── analytics.py        # Zone detection, line crossing, speed, alerts
├── output/                 # Saved recordings and screenshots
├── models/                 # Downloaded YOLO weights (auto-managed)
├── requirements.txt
└── README.md
```

---

## 🧠 Model Comparison

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| `yolov8n` | 3.2 M | ⚡ Fastest | Good | Real-time on CPU / laptop |
| `yolov8s` | 11.2 M | ⚡ Fast | Better | Balanced performance |
| `yolov8m` | 25.9 M | 🔄 Moderate | Best | High accuracy with GPU |

---

## 🛠️ Tech Stack

- **Detection:** Ultralytics YOLOv8
- **Tracking:** ByteTrack (built-in via Ultralytics)
- **Video Processing:** OpenCV
- **UI:** Streamlit
- **Compute:** PyTorch (CPU / CUDA auto-detection)
- **Analytics:** NumPy, Pandas

---

## 📸 Output Examples

All recordings and screenshots are saved to the `output/` directory with timestamped filenames.

---

## 📜 License

This project is for educational and research purposes.
