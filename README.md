# 🌾 AI Grain Counter (VisionOS v1.2.0)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.0-00FF00.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/anandmahadev/grain.detector/graphs/commit-activity)

A high-performance computer vision application for automated grain detection, counting, and classification. Leveraging **YOLOv8** and **OpenCV**, this system provides real-time analytical insights for agricultural seed quality control with a focus on ease of use and professional reporting.

## 📖 Table of Contents
- [✨ Core Features](#-core-features)
- [🏗 System Architecture](#-system-architecture)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [⚙️ Configuration](#-configuration)
- [🧪 Testing Suite](#-testing-suite)
- [🛡 Security & Compliance](#-security--compliance)
- [🤝 Contributing](#-contributing)

## ✨ Core Features
- **Dual Detection Engine**: Toggle between high-precision Watershed (OpenCV) and intelligent Object Detection (YOLOv8).
- **Real-Time WebRTC**: Process live video streams with ultra-low latency directly in the browser.
- **Analytical Dashboard**: Live Plotly charts displaying count breakdowns and detection distributions.
- **Export Ready**: One-click CSV export for downstream data analysis and reporting.
- **Dark Mode Optimized**: Premium UI with glassmorphism aesthetics and responsive design.

## 🏗 System Architecture
```mermaid
graph TD
    User([User]) --> UI[Streamlit Interface]
    UI --> Input{Input Source}
    Input -->|Upload| Static[Static Image Processor]
    Input -->|WebRTC| Live[Real-time Video Processor]
    
    Static --> Engine[Core AI Engine]
    Live --> Engine
    
    Engine --> CV[OpenCV Pipeline]
    Engine --> YOLO[YOLOv8 Engine]
    
    CV --> Watershed[Watershed Algorithm]
    YOLO --> Neuro[Neural Weights .pt]
    
    Watershed --> Dash[Analytics Dashboard]
    Neuro --> Dash
    
    Dash --> Export[CSV/PDF Reports]
```

## 🚀 Quick Start

### Automatic Installation (Recommended)
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/anandmahadev/grain.detector.git
   cd grain.detector
   ```
2. **Execute Automation Script:**
   ```bash
   run_all.bat
   ```
   *This script automates environment creation, dependency resolution, and launches the application.*

### Manual Setup
```bash
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
streamlit run grain_counter.py
```

## 🧪 Testing Suite
Maintain system integrity with our automated test suite, which now includes static asset validation and core logic checks:
```bash
pytest tests/
```

## ⚙️ Troubleshooting
- **GPU Acceleration**: If you have an NVIDIA GPU, ensure CUDA is installed for faster YOLO processing.
- **Camera Access**: If the webcam mode fails, ensure your browser has permissions and no other app is using the camera.
- **Dependency Issues**: Try clearing your environment and re-running `pip install -r requirements.txt`.

## 🛡 Security & Compliance
We follow strict development standards. Please refer to our [SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more details.

## 🤝 Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. 

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
Developed with ❤️ by **Anand Mahadev**
