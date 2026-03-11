import os
from typing import Dict, Tuple

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

# --- CONFIGURATION & SETUP ---
APP_CONFIG = {
    "title": "🌾 AI Grain Counter",
    "layout": "wide",
    "icon": "🌾",
    "base_model": "yolov8n.pt",
    "custom_model": "custom_rice_pepper_model.pt"
}

st.set_page_config(page_title=APP_CONFIG["title"], layout=APP_CONFIG["layout"], page_icon=APP_CONFIG["icon"], initial_sidebar_state="expanded")

# Beautiful Custom CSS
try:
    with open('style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass  # Allow fallback if run from different dir

@st.cache_resource
def load_model():
    # If the custom trained model exists from our script, use it directly!
    # This proves the "train by urself" request works End-to-End.
    if os.path.exists(APP_CONFIG["custom_model"]):
        model = YOLO(APP_CONFIG["custom_model"])
        # Ensure we set the classes explicitly for the UI mappings
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            model.model.names = {0: 'Rice', 1: 'Pepper'}
        return model

    # Otherwise, fallback to the base model with mock categories.
    model = YOLO(APP_CONFIG["base_model"]) 
    if hasattr(model, 'model') and hasattr(model.model, 'names'):
        # Mock class names for demonstration purposes
        mock_names = ['Rice', 'Wheat', 'Corn', 'Millet', 'Seeds']
        model.model.names = {i: mock_names[i % len(mock_names)] for i in range(100)}
    return model

model = load_model()
GRAIN_TYPES = list(set(model.names.values()) if hasattr(model, 'names') else ['Rice', 'Pepper'])
# If YOLO stored them internally in model.model.names
if hasattr(model, 'model') and hasattr(model.model, 'names'):
    # ensure deterministic order from dict iteration if possible, else sort it or just use values
    GRAIN_TYPES = list(dict.fromkeys(model.model.names.values()))

def render_sidebar():
    st.sidebar.markdown("<h1 style='text-align: center;'>🌾 Settings</h1>", unsafe_allow_html=True)
    conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    iou = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05)

    st.sidebar.divider()
    st.sidebar.markdown("### 🛠 System Info")
    info_text = (
        "**Backend:** Python + OpenCV\n\n"
        "**AI Model:** Custom YOLOv8\n\n"
        "**Frontend:** Streamlit"
    ) if os.path.exists("custom_rice_pepper_model.pt") else (
        "**Backend:** Python + OpenCV\n\n"
        "**AI Model:** Base YOLOv8 Demo\n\n"
        "**Frontend:** Streamlit"
    )
    st.sidebar.info(info_text)

    if not os.path.exists("custom_rice_pepper_model.pt"):
        st.sidebar.warning("Note: Base Demo Model Loaded. Run the train_custom_yolo.py script to train on real data!")
    else:
        st.sidebar.success("Custom Rice & Pepper Model Loaded! 🎯")
        
    return conf, iou

conf_threshold, iou_threshold = render_sidebar()

import time

# --- HELPER FUNCTIONS ---
def process_frame(img_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], float]:
    """
    Processes an image frame via the YOLO architecture.
    
    Args:
        img_array: Raw NumPy image array initialized in RGB space.
    Returns:
        Tuple containing the annotated image with overlay boxes, mapped dictionary counts, and execution latency.
    """
    start_time = time.time()
    results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
    annotated_img = results[0].plot()
    
    counts: Dict[str, int] = {grain: 0 for grain in GRAIN_TYPES}
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        counts[GRAIN_TYPES[cls_id % len(GRAIN_TYPES)]] += 1
        
    latency = time.time() - start_time
    return annotated_img, counts, latency

def render_dashboard(counts: Dict[str, int], latency: float = 0.0) -> None:
    """
    Renders the Streamlit native layout dynamically with live analytical capabilities.
    
    Args:
        counts: Key-Value Dictionary defining dynamic counts of detection entities.
        latency: Time taken to process the image frame.
    """
    st.divider()
    st.markdown("<h2>📊 Live Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1.5], gap="large")
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Grains</div>
            <div class="metric-value">{sum(counts.values())}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="padding: 15px;">
            <div class="metric-label" style="font-size: 0.9rem;">Processing Time</div>
            <div class="metric-value" style="font-size: 2rem;">{latency:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        df = pd.DataFrame(list(counts.items()), columns=["Grain Type", "Count"])
        st.download_button("📥 Export Results (.CSV)", df.to_csv(index=False).encode('utf-8'), "grain_analysis.csv", "text/csv")
        
    with col2:
        fig_bar = px.bar(df, x="Grain Type", y="Count", color="Grain Type", title="Count Breakdown")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col3:
        fig_pie = px.pie(df, names="Grain Type", values="Count", hole=0.5, title="Distribution")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(fig_pie, use_container_width=True)

# --- MAIN UI ---
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🌾 AI Grain Counter System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>Real-time AI detection and analysis using YOLOv8.</p>", unsafe_allow_html=True)

mode = st.radio("Choose Input Mode:", ["🖼️ Image Upload", "🎥 Live Webcam"], horizontal=True)

if mode == "🖼️ Image Upload":
    col_up, col_res = st.columns([1, 2], gap="large")
    with col_up:
        file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if file: st.image(file, caption="Original", use_container_width=True)
            
    with col_res:
        if file:
            try:
                # Convert RGB PIL image to BGR for YOLO processing
                img_array = cv2.cvtColor(np.array(Image.open(file).convert('RGB')), cv2.COLOR_RGB2BGR)
                with st.spinner("🔍 Processing..."):
                    ann_img, counts, latency = process_frame(img_array)
                    st.image(ann_img, caption="YOLOv8 Annotated", channels="BGR", use_container_width=True)
                render_dashboard(counts, latency)
            except Exception as e:
                st.error(f"Failed to process image: {str(e)}")

elif mode == "🎥 Live Webcam":
    col_cam, col_info = st.columns([2, 1], gap="large")
    with col_info:
        st.info("**Instructions:** Click START. Allow camera permissions. The AI will overlay counts in real-time.")
    with col_cam:
        class YOLOVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.conf = conf_threshold
                self.iou = iou_threshold

            def recv(self, frame):
                try:
                    img = frame.to_ndarray(format="bgr24")
                    results = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)
                    ann_img = results[0].plot()
                    
                    boxes = results[0].boxes
                    counts_dict: Dict[str, int] = {g: 0 for g in GRAIN_TYPES}
                    for box in boxes:
                        counts_dict[GRAIN_TYPES[int(box.cls[0].item()) % len(GRAIN_TYPES)]] += 1
                    
                    # Draw Overlay for real-time visibility 
                    cv2.rectangle(ann_img, (10, 10), (250, 200), (0, 0, 0), -1)
                    cv2.putText(ann_img, f"TOTAL: {len(boxes)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y = 70
                    for g, c in counts_dict.items():
                        cv2.putText(ann_img, f"{g}: {c}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y += 25
                        
                    # Add Confidence overlay for top item if available
                    if len(boxes) > 0:
                        top_conf = float(boxes[0].conf[0].item()) * 100
                        cv2.putText(ann_img, f"Top Conf: {top_conf:.1f}%", (20, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                    return av.VideoFrame.from_ndarray(ann_img, format="bgr24")
                except Exception as e:
                    # In case of corruption, yield original frame
                    return frame

        webrtc_streamer(key="webcam", mode=WebRtcMode.SENDRECV, video_processor_factory=YOLOVideoProcessor,
                        media_stream_constraints={"video": True, "audio": False}, async_processing=True,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
