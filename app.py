import streamlit as st
import time
import random
from PIL import Image

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="MonFresh - Kiểm tra độ tươi thực phẩm",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS TOÀN DIỆN ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --primary: #10B981;
        --primary-dark: #059669;
        --primary-light: #D1FAE5;
        --text-main: #064E3B;
        --text-muted: #6B7280;
        --bg-app: #F0FDF4;
        --card-bg: #FFFFFF;
        --border: #E5E7EB;
        --danger: #EF4444;
        --warning: #F59E0B;
    }

    * { box-sizing: border-box; }
    
    body {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: var(--bg-app);
        color: var(--text-main);
        margin: 0;
        padding: 0;
    }

    /* Ẩn các element mặc định của Streamlit */
    #MainMenu, footer, header {visibility: hidden !important;}
    .stApp > header {display: none !important;}
    
    /* Layout chính */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 40px 0;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 42px;
        font-weight: 800;
        color: var(--primary-dark);
        margin: 0;
        letter-spacing: -1px;
    }
    .app-subtitle {
        font-size: 16px;
        color: var(--text-muted);
        margin-top: 8px;
    }

    /* Grid Layout */
    .content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        align-items: start;
    }

    /* Cards */
    .card {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* Custom Tabs - KHÔNG DÙNG RADIO */
    .tabs-container {
        display: flex;
        background: #F3F4F6;
        padding: 5px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .tab-option {
        flex: 1;
        text-align: center;
        padding: 12px;
        font-weight: 600;
        font-size: 14px;
        color: var(--text-muted);
        cursor: pointer;
        border-radius: 8px;
        transition: all 0.2s;
        user-select: none;
    }
    .tab-option.active {
        background: white;
        color: var(--primary-dark);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Upload Area */
    .upload-area {
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        background: #F8FAFC;
        transition: all 0.3s;
        cursor: pointer;
    }
    .upload-area:hover {
        border-color: var(--primary);
        background: var(--primary-light);
    }
    .upload-icon {
        font-size: 48px;
        margin-bottom: 15px;
        opacity: 0.6;
    }

    /* Nút bấm */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        border: none;
        padding: 16px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        transition: transform 0.2s;
        margin-top: 20px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }

    /* Kết quả */
    .result-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .badge-fresh { background: #DCFCE7; color: #166534; }
    .badge-warning { background: #FEF3C7; color: #92400E; }
    .badge-bad { background: #FEE2E2; color: #991B1B; }

    .score-display {
        text-align: center;
        margin: 30px 0;
    }
    .score-number {
        font-size: 64px;
        font-weight: 800;
        color: var(--primary-dark);
        line-height: 1;
    }
    .score-label {
        font-size: 14px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 15px 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: var(--text-muted); font-size: 14px; }
    .info-value { font-weight: 600; color: var(--text-main); font-size: 15px; }

    .recommendation-box {
        background: #ECFDF5;
        border-left: 4px solid var(--primary);
        padding: 20px;
        border-radius: 8px;
        margin-top: 25px;
        color: var(--text-main);
        line-height: 1.6;
    }

    /* Ẩn hoàn toàn radio widget gốc */
    .stRadio {
        visibility: hidden;
        position: absolute;
        width: 0;
        height: 0;
        overflow: hidden;
    }
    .stRadio > div { display: none !important; }
    
    /* Fix file uploader */
    .stFileUploader { margin: 0; }
    .stFileUploader > div { border: none !important; background: transparent !important; box-shadow: none !important; }
    .stFileUploader label { display: none; }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--text-muted);
    }
    .empty-icon { font-size: 60px; margin-bottom: 20px; opacity: 0.3; }
    .empty-title { font-size: 20px; font-weight: 700; color: var(--text-main); margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- LOGIC ---
def analyze_image(img):
    time.sleep(1.2)
    status = random.choice(['fresh', 'warning', 'bad'])
    
    if status == 'fresh':
        return {
            'score': random.randint(88, 98),
            'label': 'TƯƠI NGON',
            'badge': 'badge-fresh',
            'color': '#10B981',
            'expiry': '5-7 ngày',
            'msg': 'Thực phẩm rất tươi, có thể sử dụng ngay hoặc bảo quản ngăn mát.'
        }
    elif status == 'warning':
        return {
            'score': random.randint(60, 75),
            'label': 'CẦN LƯU Ý',
            'badge': 'badge-warning',
            'color': '#F59E0B',
            'expiry': '1-2 ngày',
            'msg': 'Thực phẩm bắt đầu giảm chất lượng. Nên chế biến sớm và nấu chín kỹ.'
        }
    else:
        return {
            'score': random.randint(20, 50),
            'label': 'KHÔNG TƯƠI',
            'badge': 'badge-bad',
            'color': '#EF4444',
            'expiry': 'Không nên dùng',
            'msg': 'Thực phẩm có dấu hiệu hư hỏng. Không nên sử dụng để đảm bảo an toàn.'
        }

# --- STATE ---
if 'mode' not in st.session_state:
    st.session_state.mode = 'upload'
if 'image' not in st.session_state:
    st.session_state.image = None
if 'result' not in st.session_state:
    st.session_state.result = None

# --- UI ---
st.markdown("""
<div class="main-container">
    <div class="app-header">
        <h1 class="app-title">🌿 MonFresh</h1>
        <p class="app-subtitle">Kiểm tra độ tươi thực phẩm bằng AI</p>
    </div>
    
    <div class="content-grid">
        <!-- CỘT TRÁI: INPUT -->
        <div class="card">
            <!-- Custom Tabs JS -->
            <div class="tabs-container">
                <div class="tab-option active" onclick="switchTab('upload')" id="tab-upload">📁 Tải ảnh lên</div>
                <div class="tab-option" onclick="switchTab('camera')" id="tab-camera">📷 Chụp ảnh</div>
            </div>
            
            <script>
                function switchTab(mode) {
                    document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
                    document.getElementById('tab-camera').classList.toggle('active', mode === 'camera');
                    // Gửi tín hiệu về Streamlit qua session state nếu cần thiết
                }
            </script>
            
            <!-- Radio ẩn để điều khiển logic -->
            mode = st.radio("Chọn chế độ", ["upload", "camera"], index=0, label_visibility="collapsed", key="radio_mode")
            st.session_state.mode = mode
            
            if mode == "upload":
                file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
                if file:
                    st.session_state.image = Image.open(file)
                    st.session_state.result = None
                    st.image(st.session_state.image, use_container_width=True)
            else:
                cam = st.camera_input("Chụp ảnh", label_visibility="collapsed")
                if cam:
                    st.session_state.image = Image.open(cam)
                    st.session_state.result = None
                    st.image(st.session_state.image, use_container_width=True)
            
            if st.session_state.image:
                if st.button("PHÂN TÍCH NGAY"):
                    with st.spinner("Đang phân tích..."):
                        st.session_state.result = analyze_image(st.session_state.image)
                        st.rerun()
        </div>
        
        <!-- CỘT PHẢI: KẾT QUẢ -->
        <div class="card">
            if st.session_state.result is None:
                st.markdown('''
                <div class="empty-state">
                    <div class="empty-icon">🥩</div>
                    <div class="empty-title">Chưa có kết quả</div>
                    <p>Vui lòng tải ảnh lên hoặc chụp ảnh để bắt đầu phân tích độ tươi.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                res = st.session_state.result
                st.markdown(f'''
                <div style="text-align: center;">
                    <span class="result-badge {res['badge']}">{res['label']}</span>
                    
                    <div class="score-display">
                        <div class="score-number" style="color: {res['color']}">{res['score']}</div>
                        <div class="score-label">Điểm chất lượng</div>
                    </div>
                    
                    <div style="text-align: left; margin-top: 30px;">
                        <div class="info-row">
                            <span class="info-label">Thời hạn sử dụng ước tính</span>
                            <span class="info-value">{res['expiry']}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Trạng thái</span>
                            <span class="info-value" style="color: {res['color']}">● {res['label']}</span>
                        </div>
                    </div>
                    
                    <div class="recommendation-box">
                        <strong>💡 Khuyến nghị:</strong><br>
                        {res['msg']}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
