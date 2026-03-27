import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from roboflow import Roboflow

# ==========================================
# ⚙️ CONFIGURATION (ส่วนการตั้งค่าหลังบ้าน)
# ==========================================
ROBOFLOW_API_KEY = "ru4RauU12e2iydHfUHEN"  # API Key ของคุณ
PROJECT_ID = "accident-uqy2q"             # Project ID
MODEL_VERSION = 3                         # เลขเวอร์ชันโมเดล
EVIDENCE_FOLDER = "evidence"              # ชื่อโฟลเดอร์เก็บรูปหลักฐาน
# ==========================================

if not os.path.exists(EVIDENCE_FOLDER):
    os.makedirs(EVIDENCE_FOLDER)

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Accident Monitoring", layout="wide", page_icon="🚨")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- ระบบหน่วยความจำ (Session State) ---
if 'total_accidents' not in st.session_state:
    st.session_state.total_accidents = 0
if 'history_logs' not in st.session_state:
    st.session_state.history_logs = []
# ตัวแปรสำหรับล็อคการนับในแต่ละรอบการรัน
if 'has_counted_this_video' not in st.session_state:
    st.session_state.has_counted_this_video = False

# --- ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_hub_model():
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace().project(PROJECT_ID)
        return project.version(MODEL_VERSION).model
    except Exception as e:
        st.error(f"❌ เชื่อมต่อ Model Hub ไม่สำเร็จ: {e}")
        return None

# --- ส่วนควบคุมด้านข้าง (SIDEBAR) ---
with st.sidebar:
    st.title("🛡️ AI Command Center")
    st.divider()
    
    source_option = st.selectbox("📌 แหล่งที่มาวิดีโอ", ["อัปโหลดไฟล์วิดีโอ", "เปิดกล้อง WebCam"])
    
    uploaded_file = None
    if source_option == "อัปโหลดไฟล์วิดีโอ":
        uploaded_file = st.file_uploader("เลือกไฟล์วิดีโอ (MP4, AVI)", type=['mp4', 'avi', 'mov'])
    
    st.divider()
    run_btn = st.button("🚀 เริ่มการวิเคราะห์", type="primary", use_container_width=True)
    
    if st.button("🔄 รีเซ็ตระบบ (ล้างการนับ)", use_container_width=True):
        st.session_state.total_accidents = 0
        st.session_state.has_counted_this_video = False
        st.session_state.history_logs = []
        for f in os.listdir(EVIDENCE_FOLDER):
            os.remove(os.path.join(EVIDENCE_FOLDER, f))
        st.rerun()

# --- หน้าจอหลัก (MAIN UI) ---
st.title("🚦 Accident Monitoring Dashboard")

m1, m2, m3 = st.columns(3)
count_metric = m1.empty()
status_metric = m2.empty()
m3.metric("System Health", "Active", "Stable")

count_metric.metric("จำนวนอุบัติเหตุ", f"{st.session_state.total_accidents} ครั้ง")
status_metric.metric("สถานะปัจจุบัน", "Ready")

tab1, tab2 = st.tabs(["📺 Live Monitoring", "📂 Evidence Gallery"])

with tab1:
    col_v, col_l = st.columns([3, 1])
    video_feed = col_v.image([], use_container_width=True)
    log_display = col_l.empty()

# --- ส่วนประมวลผลวิดีโอ ---
if run_btn:
    model = load_hub_model()
    if model:
        # รีเซ็ตสถานะการนับใหม่ทุกครั้งที่กดปุ่มเริ่มวิเคราะห์วิดีโอใหม่
        st.session_state.has_counted_this_video = False
        
        video_source = 0
        if source_option == "อัปโหลดไฟล์วิดีโอ":
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                video_source = tfile.name
            else:
                st.warning("⚠️ กรุณาเลือกไฟล์วิดีโอก่อน")
                st.stop()
        
        cap = cv2.VideoCapture(video_source)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # ส่งภาพไปวิเคราะห์
            prediction = model.predict(frame, confidence=45, overlap=30).json()
            predictions = prediction['predictions']
            
            if len(predictions) > 0:
                status_metric.error("🔴 ตรวจพบอุบัติเหตุ!")
                
                # --- LOGIC: นับครั้งเดียวต่อคลิป ---
                if not st.session_state.has_counted_this_video:
                    st.session_state.total_accidents += 1
                    st.session_state.has_counted_this_video = True # ล็อคทันทีเพื่อไม่ให้นับซ้ำ
                    
                    now = datetime.now()
                    img_name = f"{EVIDENCE_FOLDER}/crash_{now.strftime('%H%M%S')}.jpg"
                    cv2.imwrite(img_name, frame)
                    
                    st.session_state.history_logs.insert(0, f"⚠️ {now.strftime('%H:%M:%S')} - พบเหตุการณ์ (บันทึกแล้ว)")
                    count_metric.metric("จำนวนอุบัติเหตุ", f"{st.session_state.total_accidents} ครั้ง", delta=1)
                
                # วาดกรอบสี่เหลี่ยมแสดงบนหน้าจอ
                for pred in predictions:
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    start_p = (int(x - w/2), int(y - h/2))
                    end_p = (int(x + w/2), int(y + h/2))
                    cv2.rectangle(frame, start_p, end_p, (255, 0, 0), 4)
            else:
                status_metric.success("✅ ปกติ")

            video_feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with log_display.container():
                for log in st.session_state.history_logs[:10]: st.write(log)

        cap.release()

# --- Gallery ---
with tab2:
    files = sorted(os.listdir(EVIDENCE_FOLDER), reverse=True)
    if files:
        cols = st.columns(4)
        for idx, f in enumerate(files[:12]):
            with cols[idx % 4]:
                st.image(f"{EVIDENCE_FOLDER}/{f}", caption=f)
    else:
        st.info("ยังไม่มีข้อมูลในคลังหลักฐาน")