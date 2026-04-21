import streamlit as st
import time
import random

# Cấu hình trang
st.set_page_config(
    page_title="MeatFresh - Kiểm tra độ tươi thịt",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --meat-red: #DC2626;
        --meat-red-light: #FEE2E2;
        --fresh-green: #059669;
        --fresh-green-light: #D1FAE5;
        --warning-orange: #D97706;
        --warning-orange-light: #FEF3C7;
        --bg-app: #FAFAFA;
        --bg-card: #FFFFFF;
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --border: #E5E7EB;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        --radius: 20px;
    }

    * { box-sizing: border-box; }

    body {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: var(--bg-app);
        color: var(--text-primary);
    }

    /* Hide Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
    }

    /* Main Layout Grid */
    .main-grid {
        display: grid;
        grid-template-columns: 380px 1fr;
        gap: 2rem;
        align-items: start;
    }

    /* Left Panel - Input */
    .input-panel {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 2rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
        position: sticky;
        top: 2rem;
    }

    /* Right Panel - Results */
    .result-panel {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    /* Header */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }
    .brand-logo {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--meat-red), #EF4444);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: white;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    .brand-title h1 {
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    .brand-title p {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0.25rem 0 0 0;
        font-weight: 500;
    }

    /* Upload Zone */
    .upload-zone {
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        background: #F9FAFB;
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 1.5rem;
    }
    .upload-zone:hover {
        border-color: var(--meat-red);
        background: var(--meat-red-light);
    }
    .upload-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        opacity: 0.6;
    }
    .upload-text {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    .upload-subtext {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    /* Mode Switcher */
    .mode-switcher {
        display: flex;
        background: #F3F4F6;
        padding: 4px;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .mode-btn {
        flex: 1;
        padding: 0.75rem;
        text-align: center;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
        border: none;
        background: transparent;
        color: var(--text-secondary);
    }
    .mode-btn.active {
        background: white;
        color: var(--meat-red);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Analyze Button */
    .analyze-btn {
        width: 100%;
        padding: 1rem;
        background: linear-gradient(135deg, var(--meat-red), #EF4444);
        color: white;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25);
    }
    .analyze-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(220, 38, 38, 0.3);
    }

    /* Result Cards */
    .result-card {
        background: var(--bg-card);
        border-radius: var(--radius);
        padding: 2rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.25rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .status-fresh {
        background: var(--fresh-green-light);
        color: var(--fresh-green);
    }
    .status-warning {
        background: var(--warning-orange-light);
        color: var(--warning-orange);
    }
    .status-spoiled {
        background: var(--meat-red-light);
        color: var(--meat-red);
    }

    /* Score Display */
    .score-display {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #F9FAFB, #FFFFFF);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }
    .score-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--meat-red), var(--fresh-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .metric-item {
        text-align: center;
        padding: 1.25rem;
        background: #F9FAFB;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    .metric-name {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
    }

    /* Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #F0FDF4, #FFFFFF);
        border: 1px solid #BBF7D0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .rec-title {
        font-weight: 700;
        color: var(--fresh-green);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .rec-text {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 0.95rem;
    }

    /* Empty State */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 5rem 2rem;
        text-align: center;
        background: var(--bg-card);
        border-radius: var(--radius);
        border: 2px dashed var(--border);
    }
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        opacity: 0.3;
    }
    .empty-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    .empty-desc {
        color: var(--text-secondary);
        max-width: 400px;
        line-height: 1.6;
    }

    /* Progress Bar */
    .progress-wrapper {
        margin-top: 1rem;
    }
    .progress-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-secondary);
    }
    .progress-track {
        height: 12px;
        background: #E5E7EB;
        border-radius: 99px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Hide Streamlit specific elements inside columns */
    .stFileUploader > div { border: none !important; padding: 0 !important; background: none !important; }
    .stCameraInput > div { border: none !important; }
    
    @media (max-width: 900px) {
        .main-grid { grid-template-columns: 1fr; }
        .input-panel { position: static; }
    }
</style>
""", unsafe_allow_html=True)

# --- APP LOGIC ---

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'result' not in st.session_state:
    st.session_state.result = None

# Main Grid Layout
col_left, col_right = st.columns([380, 1], gap="small")

with col_left:
    # Brand Header
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🥩</div>
        <div class="brand-title">
            <h1>MeatFresh</h1>
            <p>AI Freshness Detector</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Switcher (Visual only, logic handled below)
    mode = st.radio(
        "Chọn chế độ",
        ["📷 Chụp ảnh", "📁 Tải lên"],
        label_visibility="collapsed",
        index=1,
        horizontal=True
    )
    
    # Upload/Camera Area
    uploaded_file = None
    camera_input = None
    
    if "📁" in mode:
        uploaded_file = st.file_uploader(
            "",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed",
            help="Kéo thả ảnh hoặc click để chọn"
        )
    else:
        camera_input = st.camera_input("Chụp ảnh", label_visibility="collapsed")
    
    active_image = uploaded_file if uploaded_file else camera_input
    
    # Display Image Preview
    if active_image:
        st.image(active_image, use_container_width=True, output_format="JPEG")
        
        if st.button("🔍 Phân tích ngay", key="analyze", type="primary"):
            with st.spinner('Đang phân tích hình ảnh...'):
                time.sleep(1.5)
            
            # Generate Mock Data
            score = random.uniform(65, 99)
            
            if score >= 90:
                status = "TƯƠI NGON"
                status_class = "status-fresh"
                color = "#059669"
                rec = "Thịt rất tươi, chất lượng tuyệt vời. Có thể bảo quản ngăn mát 3-5 ngày hoặc sử dụng ngay."
            elif score >= 75:
                status = "CẦN LƯU Ý"
                status_class = "status-warning"
                color = "#D97706"
                rec = "Thịt còn dùng được nhưng nên chế biến trong hôm nay. Không nên bảo quản lâu dài."
            else:
                status = "KHÔNG TƯƠI"
                status_class = "status-spoiled"
                color = "#DC2626"
                rec = "Cảnh báo: Thịt có dấu hiệu hư hỏng. Khuyến nghị KHÔNG nên sử dụng để đảm bảo an toàn."
            
            st.session_state.result = {
                'score': score,
                'status': status,
                'class': status_class,
                'color': color,
                'rec': rec,
                'metrics': {
                    'màu sắc': random.uniform(80, 99),
                    'kết cấu': random.uniform(75, 98),
                    'độ ẩm': random.uniform(70, 95)
                }
            }
            st.session_state.analyzed = True
            st.rerun()
    
    elif st.session_state.analyzed:
        # Keep showing last image if available (simplified for demo)
        pass
    else:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">📤</div>
            <div class="upload-text">Chưa chọn ảnh</div>
            <div class="upload-subtext">Vui lòng tải ảnh hoặc chụp ảnh thịt cần kiểm tra</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if st.session_state.analyzed and st.session_state.result:
        res = st.session_state.result
        
        # Main Result Card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header">
                <div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem; font-weight: 600;">KẾT QUẢ PHÂN TÍCH</div>
                    <div style="font-size: 2rem; font-weight: 800; color: var(--text-primary); line-height: 1.2;">{res['status']}</div>
                </div>
                <div class="status-badge {res['class']}">
                    {res['status']}
                </div>
            </div>
            
            <div class="score-display">
                <div class="score-value">{res['score']:.1f}%</div>
                <div class="score-label">Chỉ số tươi ngon</div>
            </div>
            
            <div class="progress-wrapper">
                <div class="progress-labels">
                    <span>Đánh giá tổng quan</span>
                    <span>{res['score']:.0f}/100</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {res['score']}%; background: {res['color']};"></div>
                </div>
            </div>
            
            <div class="metrics-grid">
        """, unsafe_allow_html=True)
        
        # Metrics
        for name, value in res['metrics'].items():
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{value:.1f}%</div>
                <div class="metric-name">{name.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recommendation
        bg_color = "#F0FDF4" if res['score'] >= 75 else "#FEF2F2"
        border_color = "#BBF7D0" if res['score'] >= 75 else "#FECACA"
        text_color = "#065F46" if res['score'] >= 75 else "#991B1B"
        
        st.markdown(f"""
        <div class="recommendation-box" style="background: {bg_color}; border-color: {border_color};">
            <div class="rec-title" style="color: {text_color};">
                💡 Khuyến nghị từ chuyên gia
            </div>
            <div class="rec-text" style="color: {text_color};">
                {res['rec']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Empty State
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-title">Chưa có kết quả phân tích</div>
            <div class="empty-desc">
                Vui lòng tải ảnh lên hoặc chụp ảnh thịt ở cột bên trái và nhấn "Phân tích ngay" để xem kết quả chi tiết.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid var(--border); color: var(--text-secondary); font-size: 0.85rem;">
    © 2024 MeatFresh AI • Công nghệ kiểm tra thực phẩm thông minh
</div>
""", unsafe_allow_html=True)
