import streamlit as st
import time
import random

# Cấu hình trang
st.set_page_config(
    page_title="MonFresh - Kiểm tra độ tươi thịt",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS với thiết kế hiện đại, loại bỏ hoàn toàn các element gây lỗi
st.markdown("""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --primary: #0F766E;       /* Teal 700 */
        --primary-light: #14B8A6; /* Teal 500 */
        --primary-dark: #115E59;  /* Teal 800 */
        --accent: #F0FDFA;        /* Teal 50 */
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --text-main: #1F2937;
        --text-muted: #6B7280;
        --bg-app: #F3F4F6;
        --white: #FFFFFF;
        --radius-lg: 24px;
        --radius-md: 16px;
        --radius-sm: 12px;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        --shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
    }

    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Reset Streamlit defaults causing issues */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hide default file uploader label and border if any */
    .stFileUploader {
        margin: 0;
        padding: 0;
    }
    .stFileUploader > div {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    .stFileUploader label {
        display: none !important;
    }
    .stFileUploader section {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    /* Main Layout */
    .main-content {
        background: var(--bg-app);
        min-height: 100vh;
        padding: 2rem;
        border-radius: var(--radius-lg);
    }

    /* Header Branding */
    .brand-header {
        margin-bottom: 2.5rem;
        text-align: left;
    }
    .brand-logo {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .brand-subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* Custom Tabs - Replacing the ugly slider */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border: none;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: var(--radius-md);
        background: var(--white);
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--accent);
        color: var(--primary);
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3);
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] button[role="tab"] {
        border: none;
    }
    /* Hide the red underline/bar completely */
    .stTabs [data-baseweb="tab"] span {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }

    /* Input Card */
    .input-card {
        background: var(--white);
        border-radius: var(--radius-lg);
        padding: 2rem;
        box-shadow: var(--shadow);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid rgba(0,0,0,0.02);
    }

    /* Custom Upload Zone */
    .upload-zone {
        border: 2px dashed #CBD5E1;
        border-radius: var(--radius-md);
        padding: 3rem 1.5rem;
        text-align: center;
        background: #F8FAFC;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .upload-zone:hover {
        border-color: var(--primary-light);
        background: var(--accent);
        transform: translateY(-2px);
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        opacity: 0.8;
    }
    .upload-title {
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .upload-desc {
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    /* Result Card */
    .result-card {
        background: var(--white);
        border-radius: var(--radius-lg);
        padding: 2.5rem;
        box-shadow: var(--shadow);
        height: 100%;
        border: 1px solid rgba(0,0,0,0.02);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
    }
    .status-fresh { background: #D1FAE5; color: #065F46; }
    .status-caution { background: #FEF3C7; color: #92400E; }
    .status-spoiled { background: #FEE2E2; color: #991B1B; }

    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .metric-item {
        background: #F8FAFC;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        text-align: center;
        border: 1px solid #E2E8F0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-main);
    }
    .metric-unit {
        font-size: 0.9rem;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
        border-left: 4px solid var(--primary);
        padding: 1.5rem;
        border-radius: var(--radius-sm);
        margin-top: 2rem;
    }
    .rec-title {
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .rec-text {
        color: var(--text-main);
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 400px;
        text-align: center;
        color: var(--text-muted);
    }
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        opacity: 0.3;
        filter: grayscale(100%);
    }
    .empty-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.5rem;
    }
    .empty-desc {
        font-size: 1rem;
        max-width: 300px;
        line-height: 1.5;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.85rem;
        color: var(--text-muted);
        padding-top: 2rem;
        border-top: 1px solid #E5E7EB;
    }
    
    /* Hide Streamlit warnings */
    .stWarning {display: none;}
</style>
""", unsafe_allow_html=True)

# --- LOGIC APPLICATION ---

def analyze_image(image):
    """Giả lập quá trình phân tích AI"""
    time.sleep(1.5) # Giả lập thời gian xử lý
    
    # Random kết quả cho demo
    freshness_score = random.uniform(85, 99)
    status = "FRESH" if freshness_score > 85 else ("CAUTION" if freshness_score > 60 else "SPOILED")
    
    return {
        "score": freshness_score,
        "status": status,
        "time": "0.8s",
        "confidence": random.uniform(92, 99),
        "recommendation": "Sản phẩm có vẻ rất tươi. Có thể sử dụng ngay hoặc bảo quản ngăn mát trong 2-3 ngày." if status == "FRESH" else "Cần kiểm tra kỹ mùi và màu sắc trước khi sử dụng."
    }

# --- GIAO DIỆN CHÍNH ---

# Header
st.markdown("""
<div class="brand-header">
    <div class="brand-logo">MonFresh</div>
    <div class="brand-subtitle">Giải pháp kiểm tra độ tươi thực phẩm bằng AI</div>
</div>
""", unsafe_allow_html=True)

# Tạo 2 cột chính
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # Custom Tabs thay thế cho Radio/Slider cũ
    tab_upload, tab_camera = st.tabs(["📁 Tải ảnh lên", "📷 Chụp ảnh"])
    
    with tab_upload:
        st.markdown("""
        <div class="upload-zone">
            <span class="upload-icon">☁️</span>
            <div class="upload-title">Kéo thả ảnh vào đây</div>
            <div class="upload-desc">hoặc click để chọn file (JPG, PNG)</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True, output_format="PNG")
            if st.button("Phân tích ngay", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích..."):
                    result = analyze_image(uploaded_file)
                    st.session_state['current_result'] = result
                    st.rerun()

    with tab_camera:
        st.markdown("""
        <div class="upload-zone">
            <span class="upload-icon">📸</span>
            <div class="upload-title">Kích hoạt Camera</div>
            <div class="upload-desc">Chụp ảnh trực tiếp sản phẩm</div>
        </div>
        """, unsafe_allow_html=True)
        
        camera_file = st.camera_input("Chụp ảnh", label_visibility="collapsed")
        
        if camera_file:
            st.image(camera_file, use_container_width=True, output_format="PNG")
            if st.button("Phân tích ngay", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích..."):
                    result = analyze_image(camera_file)
                    st.session_state['current_result'] = result
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    if 'current_result' in st.session_state:
        res = st.session_state['current_result']
        
        # Xác định class màu sắc
        status_class = "status-fresh" if res['status'] == "FRESH" else ("status-caution" if res['status'] == "CAUTION" else "status-spoiled")
        status_label = "Tươi Ngon" if res['status'] == "FRESH" else ("Cần Lưu Ý" if res['status'] == "CAUTION" else "Hư Hỏng")
        icon = "✅" if res['status'] == "FRESH" else ("⚠️" if res['status'] == "CAUTION" else "❌")
        
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
            <h2 style="margin:0; color:var(--text-main);">Kết quả phân tích</h2>
            <span class="status-badge {status_class}">{icon} {status_label}</span>
        </div>
        
        <div style="font-size: 3.5rem; font-weight: 800; color: var(--primary); line-height: 1;">
            {res['score']:.1f}<span style="font-size: 1.5rem; color: var(--text-muted);">/100</span>
        </div>
        <div style="color: var(--text-muted); margin-bottom: 2rem;">Độ tin cậy: {res['confidence']:.1f}%</div>
        
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-label">Trạng thái</div>
                <div class="metric-value" style="color: {'var(--success)' if res['status']=='FRESH' else 'var(--warning)'}">{res['status']}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Thời gian</div>
                <div class="metric-value">{res['time']}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Mô hình</div>
                <div class="metric-value" style="font-size:1.2rem">AI v2.0</div>
            </div>
        </div>
        
        <div class="recommendation-box">
            <div class="rec-title">💡 Khuyến nghị từ MonFresh</div>
            <div class="rec-text">{res['recommendation']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🥩</div>
            <div class="empty-title">Chưa có dữ liệu phân tích</div>
            <div class="empty-desc">Vui lòng tải ảnh lên hoặc chụp ảnh ở cột bên trái để bắt đầu kiểm tra độ tươi.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    © 2024 MonFresh Technology. Bảo vệ sức khỏe cộng đồng.
</div>
""", unsafe_allow_html=True)
