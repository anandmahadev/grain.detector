# 🌾 AI Grain Counter (VisionOS v1.2.0)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.0-green.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance computer vision application for automated grain detection, counting, and classification. Leveraging **YOLOv8** and **OpenCV**, this system provides real-time analytical insights for agricultural seed quality control.

## ✨ Core Features
- **Dual Detection Engine**: Toggle between high-precision Watershed (OpenCV) and intelligent Object Detection (YOLOv8).
- **Real-Time WebRTC**: Process live video streams with ultra-low latency directly in the browser.
- **Analytical Dashboard**: Live Plotly charts displaying count breakdowns and detection distributions.
- **Export Ready**: One-click CSV export for downstream data analysis and reporting.
- **Dark Mode Optimized**: Premium UI with glassmorphism aesthetics and responsive design.

## 🏗 System Architecture
```mermaid
graph LR
    A[Input Source] --> B{Streamlit UI}
    B --> C[OpenCV Pipeline]
    B --> D[YOLOv8 Engine]
    C --> E[Watershed Algorithm]
    D --> F[Neuro-Weights (PT)]
    E --> G[Visual Overlay]
    F --> G[Visual Overlay]
    G --> H[Analytical Dashboard]
```

## 🚀 Quick Start

### Standard Installation
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/anandmahadev/grain.detector.git
   cd grain.detector
   ```
2. **Execute Automation Script:**
   ```bash
   run_all.bat
   ```
   *This will handle environment setup, dependency installation, and launch the application.*

### Manual Setup
```bash
pip install -r requirements.txt
streamlit run grain_counter.py
```

## 🧪 Testing Suite
Maintain system integrity with our automated test suite:
```bash
pytest tests/
```

## 🛡 Security & Compliance
We follow strict development standards. Please refer to our [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more details.

---
Developed with ❤️ by **Anand Mahadev**
