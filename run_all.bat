@echo off
echo ===================================================
echo 🌾 Installing all required dependencies...
echo ===================================================
pip install streamlit ultralytics opencv-python pillow numpy pandas plotly av streamlit-webrtc

echo.
echo ===================================================
echo 🚀 Running Custom YOLO Training Pipeline...
echo ===================================================
python train_custom_yolo.py

echo.
echo ===================================================
echo 🌟 Starting the AI Grain Counter Web App...
echo ===================================================
streamlit run grain_counter.py
pause
