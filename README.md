# AI Grain Counter 🌾

A fully functional web application that uses YOLOv8 to automatically detect, count, and classify different grains (Rice, Pepper, etc.) in real-time or via image uploads.

## Features
- **Real-time WebRTC Video Tracking**
- **Static Image Upload Support**
- **Custom Trained YOLOv8 Nano Data Integration**
- **Live Analytical Dashboard with Plotly**
- **CSV Data Export**

## Installation
Run the automatic install sequence:
```bash
run_all.bat
```
Alternatively, manually install via pip:
```bash
pip install -r requirements.txt
streamlit run grain_counter.py
```

## Running Tests
Run the established suite via Pytest:
```bash
pip install pytest
pytest tests/
```

## 🚀 Deployment

The absolute best place to deploy this app for **free** is **Streamlit Community Cloud**. Since your code is already pushed to GitHub, it takes about 2 minutes to get it live worldwide:

1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Fill in the details:
   - **Repository**: `anandmahadev/grain.detector`
   - **Branch**: `main`
   - **Main file path**: `grain_counter.py`
4. Click **"Deploy!"**

Streamlit will automatically read your `requirements.txt` file, install YOLOv8 and all dependencies, and host your AI Grain Counter on a public URL!

*(Alternative: You can also deploy this exact repository to [Hugging Face Spaces](https://huggingface.co/spaces) using the Streamlit space template)*
