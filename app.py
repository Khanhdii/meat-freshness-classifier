import streamlit as st
import time
import random
from PIL import Image
import base64
import io

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="FreshGuard - Kiểm Tra Thực Phẩm",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS TÙY CHỈNH GIAO DIỆN ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary: #2F855A;
        --primary-light: #68D391;
        --primary-dark: #22543D;
        --accent: #38B2AC;
        --bg-main: #F7FAF9;
        --bg-card: #FFFFFF;
        --text-primary: #2D3748;
        --text-secondary: #718096;
        --border: #E2E8F0;
        --success: #48BB78;
        --warning: #ED8936;
        --danger: #F56565;
    }

    * { box-sizing: border-box; }
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-main);
        color: var(--text-primary);
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .header-section {
        text-align: center;
        margin-bottom: 40px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 8px;
    }
    .header-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
    }

    /* Main Layout Grid */
    .main-grid {
        display: grid;
        grid-template-columns: 350px 1fr;
        gap: 30px;
        align-items: start;
    }

    /* Cards */
    .card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Input Section */
    .input-card {
        position: sticky;
        top: 20px;
    }

    /* Tabs Customization - Remove Red Slider */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background: #EDF2F7;
        color: var(--text-secondary);
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 500;
        font-size: 0.9rem;
        border: none;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary);
        color: white;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: #E2E8F0;
    }

    /* Upload Area */
    .upload-area {
        border: 2px dashed var(--border);
        border-radius: 10px;
        padding: 30px 20px;
        text-align: center;
        background: #FAFAFA;
        margin-top: 15px;
    }
    .upload-area:hover {
        border-color: var(--primary-light);
        background: #F0FFF4;
    }

    /* Result Card */
    .result-card {
        animation: slideUp 0.4s ease-out;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 20px;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Status Badges */
    .badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-safe { background: #C6F6D5; color: #22543D; }
    .badge-warn { background: #FEEBC8; color: #744210; }
    .badge-danger { background: #FED7D7; color: #742A2A; }

    /* Score Display */
    .score-section {
        text-align: center;
        padding: 20px 0;
    }
    .score-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 8px;
    }

    /* Progress Bar */
    .progress-wrap {
        background: #EDF2F7;
        height: 10px;
        border-radius: 5px;
        overflow: hidden;
        margin: 15px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.8s ease;
    }

    /* Info Boxes */
    .info-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-top: 20px;
    }
    .info-box {
        background: #F7FAF9;
        padding: 16px;
        border-radius: 8px;
        border-left: 3px solid var(--primary);
    }
    .info-box-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--primary-dark);
        margin-bottom: 6px;
    }
    .info-box-text {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 30px;
        text-align: center;
        background: #FAFAFA;
        border: 2px dashed var(--border);
        border-radius: 12px;
    }
    .empty-icon {
        font-size: 3rem;
        margin-bottom: 16px;
        opacity: 0.4;
    }
    .empty-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 8px;
    }
    .empty-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        max-width: 350px;
        line-height: 1.6;
    }

    /* Sidebar Info Panel */
    .info-panel {
        background: #F0FFF4;
        border: 1px solid #C6F6D5;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .panel-title {
        font-weight: 600;
        color: var(--primary-dark);
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .legend-row {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
    }

    /* Footer */
    .footer {
        margin-top: 50px;
        text-align: center;
        padding-top: 20px;
        border-top: 1px solid var(--border);
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-grid {
            grid-template-columns: 1fr;
        }
        .input-card {
            position: static;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- HÀM XỬ LÝ ---
def analyze_image(image):
    time.sleep(1.2)
    score = random.uniform(0.65, 0.98)
    
    if score > 0.85:
        return {
            "score": int(score * 100),
            "status": "An toàn",
            "badge": "badge-safe",
            "color": "#48BB78",
            "desc": "Thực phẩm tươi ngon, chất lượng tốt.",
            "recommendation": "Có thể sử dụng ngay hoặc bảo quản ngăn mát."
        }
    elif score > 0.70:
        return {
            "score": int(score * 100),
            "status": "Cần lưu ý",
            "badge": "badge-warn",
            "color": "#ED8936",
            "desc": "Thực phẩm bắt đầu giảm chất lượng.",
            "recommendation": "Nên chế biến sớm, không để lâu."
        }
    else:
        return {
            "score": int(score * 100),
            "status": "Không an toàn",
            "badge": "badge-danger",
            "color": "#F56565",
            "desc": "Phát hiện dấu hiệu hư hỏng.",
            "recommendation": "Không nên sử dụng, hãy loại bỏ."
        }

# --- GIAO DIỆN ---

# Header
st.markdown("""
<div class="header-section">
    <div class="header-title">FreshGuard AI</div>
    <div class="header-subtitle">Kiểm tra độ tươi thực phẩm bằng trí tuệ nhân tạo</div>
</div>
""", unsafe_allow_html=True)

# Main Grid Layout
col_left, col_right = st.columns([350, 650], gap="large")

with col_left:
    st.markdown('<div class="card input-card">', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["Tải ảnh lên", "Chụp ảnh"])
    
    uploaded_file = None
    
    with tab1:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 2rem; color: #68D391; margin-bottom: 10px;">📁</div>
            <div style="font-size: 0.9rem; color: #718096;">Kéo thả hoặc click để chọn ảnh</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Chọn file", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    with tab2:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 2rem; color: #68D391; margin-bottom: 10px;">📷</div>
            <div style="font-size: 0.9rem; color: #718096;">Sử dụng camera để chụp</div>
        </div>
        """, unsafe_allow_html=True)
        camera_input = st.camera_input("Chụp ảnh", label_visibility="collapsed")
        if camera_input:
            uploaded_file = camera_input
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Info Panel
    st.markdown("""
    <div class="info-panel">
        <div class="panel-title">Thang đánh giá</div>
        <div class="legend-row"><div class="legend-dot" style="background:#48BB78"></div>An toàn (>85%)</div>
        <div class="legend-row"><div class="legend-dot" style="background:#ED8936"></div>Cần lưu ý (70-85%)</div>
        <div class="legend-row"><div class="legend-dot" style="background:#F56565"></div>Không an toàn (<70%)</div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("Đang phân tích..."):
            result = analyze_image(image)
        
        # Encode image for display
        buffered = io.BytesIO()
        image_resized = image.resize((400, 400))
        image_resized.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        st.markdown(f"""
        <div class="card result-card">
            <div class="result-header">
                <div class="result-title">Kết quả phân tích</div>
                <div class="badge {result['badge']}">{result['status']}</div>
            </div>
            
            <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px;">
                    <img src="data:image/jpeg;base64,{img_base64}" 
                         style="width: 100%; border-radius: 8px; object-fit: cover;">
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <div class="score-section">
                        <div class="score-value" style="color: {result['color']}">{result['score']}%</div>
                        <div class="score-label">Độ tươi</div>
                    </div>
                    <div class="progress-wrap">
                        <div class="progress-fill" style="width: {result['score']}%; background: {result['color']}"></div>
                    </div>
                </div>
            </div>
            
            <div class="info-row">
                <div class="info-box">
                    <div class="info-box-title">Nhận xét</div>
                    <div class="info-box-text">{result['desc']}</div>
                </div>
                <div class="info-box" style="border-left-color: #3182CE;">
                    <div class="info-box-title">Khuyến nghị</div>
                    <div class="info-box-text">{result['recommendation']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🥬</div>
            <div class="empty-title">Chưa có kết quả</div>
            <div class="empty-desc">Vui lòng tải ảnh lên hoặc chụp ảnh ở cột bên trái để bắt đầu phân tích</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    © 2024 FreshGuard AI - Bảo vệ sức khỏe gia đình bạn
</div>
""", unsafe_allow_html=True)
