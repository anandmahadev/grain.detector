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
from src.engine import count_grains_opencv, APP_CONFIG, load_model

st.set_page_config(page_title=APP_CONFIG["title"], layout=APP_CONFIG["layout"], page_icon=APP_CONFIG["icon"], initial_sidebar_state="expanded")

# Beautiful Custom CSS
try:
    with open('style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass  # Allow fallback if run from different dir

model = load_model()
GRAIN_TYPES = list(dict.fromkeys(model.names.values())) if hasattr(model, 'names') else ['Rice', 'Pepper']
LABEL_STYLE = {"color": (0, 255, 0), "thickness": 2, "font_scale": 0.8}

def render_sidebar() -> Tuple[float, float, str, str]:
    """
    Renders the sidebar controls and configuration settings.
    
    Returns:
        tuple: (confidence_threshold, iou_threshold, selected_engine, roboflow_api_key)
    """
    st.sidebar.markdown("<div class='sidebar-header'>🌾 Settings</div>", unsafe_allow_html=True)
    # Allows selection between fast OpenCV methods and powerful YOLO AI models.
    engine = st.sidebar.radio("Detection Engine", ["High-Precision (OpenCV)", "YOLO AI (Local)", "Roboflow API (Cloud)"], index=0)
    
    conf = st.sidebar.slider("Confidence / Sensitivity", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("IoU / Separation", 0.05, 0.95, 0.45, 0.05)

    api_key = ""
    if engine == "Roboflow API (Cloud)":
        api_key = st.sidebar.text_input("Roboflow API Key", type="password", help="Get your key at roboflow.com")

    st.sidebar.divider()
    st.sidebar.markdown("### 🛠 System Info")
    info_text = f"**Engine:** {engine}\n\n**Backend:** OpenCV + Python\n\n**Version:** {APP_CONFIG['version']}\n\n**Developer:** {APP_CONFIG['developer']}"
    st.sidebar.info(info_text)

    with st.sidebar.expander("💡 Detection Tips"):
        st.markdown("""
        - **Lighting**: Ensure even lighting for the best results.
        - **Contrast**: Use a contrasting background (e.g., dark for rice).
        - **Separation**: Minimize grain overlapping for precision results.
        """)

    if engine == "YOLO AI (Local)" and not os.path.exists("custom_rice_pepper_model.pt"):
        st.sidebar.warning("Note: Base YOLO Demo Model Loaded. Using OpenCV mode is recommended for exact grain counting!")
        
    return conf, iou, engine, api_key

conf_threshold, iou_threshold, selected_engine, r_api_key = render_sidebar()

import time

# --- CORE LOGIC HANDLED BY SRC.ENGINE ---


# --- HELPER FUNCTIONS ---
def process_frame(img_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], float]:
    """
    Main processing pipeline for image frames. Handles selection between 
    OpenCV, Roboflow API, and Local YOLO engines.
    
    Args:
        img_array: The input image as a BGR numpy array.
        
    Returns:
        A tuple containing (annotated_image, grain_counts, processing_latency).
    """
    start_time = time.time()
    
    if selected_engine == "High-Precision (OpenCV)":
        result = count_grains_opencv(img_array, conf_threshold)
        ann_img = result.annotated_image
        counts = result.counts
        
    elif selected_engine == "Roboflow API (Cloud)":
        if not r_api_key:
            return img_array, {"Error": 0}, 0.0
        try:
            from inference_sdk import InferenceHTTPClient
            CLIENT = InferenceHTTPClient(base_url="https://detect.roboflow.com", api_key=r_api_key)
            # Using a generic public rice counting model
            result = CLIENT.infer(img_array, model_id="counting-rice-grains/1")
            
            # Simple drawing for Roboflow result
            ann_img = img_array.copy()
            total = 0
            for pred in result["predictions"]:
                x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                cv2.rectangle(ann_img, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 255), 2)
                total += 1
            counts = {"Rice/Grains": total}
        except Exception as e:
            st.error(f"Roboflow API connection error. Please verify your API Key and internet connectivity. Details: {str(e)}")
            return img_array, {"API Error": 0}, 0.0
            
    else: # YOLO AI (Local)
        results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
        ann_img = results[0].plot()
        counts = {grain: 0 for grain in GRAIN_TYPES}
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            counts[GRAIN_TYPES[cls_id % len(GRAIN_TYPES)]] += 1
        
    latency = time.time() - start_time
    return ann_img, counts, latency

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
        fig_bar = px.bar(df, x="Grain Type", y="Count", color="Count", 
                         title="Count Breakdown", color_continuous_scale="Viridis")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                              font=dict(color="#f8fafc"), margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col3:
        fig_pie = px.pie(df, names="Grain Type", values="Count", hole=0.6, 
                         title="Detection Distribution")
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"),
                               margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

# --- UI ARCHITECTURE ---
# The application is structured into two primary input modes:
# 1. 🖼️ Image Upload: For precision batch processing of static photos.
# 2. 🎥 Live Webcam: For real-time inventory checks and field analysis.
# Both modes interface with the core AI engine in src/engine.py.

# --- MAIN UI ---
st.markdown("<h1 class='main-title'>🌾 AI Grain Counter</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Precision AI analysis for Agriculture & Seed Quality Control</p>", unsafe_allow_html=True)

# Mode Selector in a nice container handled by CSS
mode = st.radio(
    "Select Input Source", 
    ["🖼️ Image Upload", "🎥 Live Real-Time Webcam"], 
    horizontal=True,
    help="Choose between processing static photos or live video streams."
)

st.divider()

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

elif mode == "🎥 Live Real-Time Webcam":
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

        webrtc_streamer(
            key="webcam", 
            mode=WebRtcMode.SENDRECV, 
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={
                "video": {
                    "facingMode": "environment"
                }, 
                "audio": False
            }, 
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

# --- FOOTER ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
    "&copy; 2026 AI Grain Counter System. Open Source Computer Vision Project.<br>"
    "Powered by Streamlit, YOLOv8, and OpenCV.</p>",
    unsafe_allow_html=True
)
# --- End of grain_counter.py ---
