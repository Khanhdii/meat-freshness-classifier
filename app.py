import streamlit as st
import time
import random

# -----------------------------------------------------------------------------
# CẤU HÌNH TRANG & GIAO DIỆN (CONFIG)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MonFresh - Kiểm Tra Độ Tươi Thực Phẩm",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ẩn các thành phần mặc định gây xấu của Streamlit
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Loại bỏ padding mặc định thừa */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CSS CUSTOM - THIẾT KẾ ĐẸP & HIỆN ĐẠI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary-color: #0F766E;       /* Teal 700 - Màu chủ đạo đậm */
        --primary-light: #14B8A6;       /* Teal 500 - Màu nhấn */
        --primary-bg: #F0FDFA;          /* Teal 50 - Nền nhẹ */
        --accent-green: #10B981;        /* Emerald 500 - Trạng thái tốt */
        --accent-warn: #F59E0B;         /* Amber 500 - Cảnh báo */
        --accent-bad: #EF4444;          /* Red 500 - Kém tươi */
        --text-main: #111827;           /* Gray 900 */
        --text-sub: #6B7280;            /* Gray 500 */
        --card-bg: #FFFFFF;
        --border-radius: 16px;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    body {
        font-family: 'Inter', sans-serif;
        background-color: #F3F4F6;
        color: var(--text-main);
    }

    /* --- HEADER --- */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--card-bg);
        padding: 1.5rem 2.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-sm);
        margin-bottom: 2rem;
        border: 1px solid rgba(0,0,0,0.03);
    }
    .brand-logo {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--primary-color);
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-logo span {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-status {
        font-size: 0.9rem;
        color: var(--text-sub);
        font-weight: 500;
        background: var(--primary-bg);
        padding: 8px 16px;
        border-radius: 20px;
        color: var(--primary-color);
    }

    /* --- LAYOUT GRID --- */
    .main-grid {
        display: grid;
        grid-template-columns: 1fr 1.2fr;
        gap: 2rem;
        align-items: start;
    }

    /* --- CARDS --- */
    .card {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 2rem;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: var(--text-main);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .card-title::before {
        content: '';
        display: block;
        width: 4px;
        height: 24px;
        background: var(--primary-light);
        border-radius: 2px;
    }

    /* --- CUSTOM TABS (NO RED SLIDER) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        color: var(--text-sub);
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white !important;
        border-color: var(--primary-color);
        box-shadow: 0 4px 6px -1px rgba(15, 118, 110, 0.3);
    }
    .stTabs [data-baseweb="tab-mark"] {
        display: none !important; /* Ẩn thanh trượt đỏ */
    }
    
    /* Input file styling */
    .stFileUploader {
        background: #F9FAFB;
        border: 2px dashed #D1D5DB;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s;
    }
    .stFileUploader:hover {
        border-color: var(--primary-light);
        background: var(--primary-bg);
    }
    .stFileUploader label {
        color: var(--text-sub);
        font-weight: 500;
    }

    /* --- RESULTS STYLING --- */
    .result-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    .badge-fresh { background: #D1FAE5; color: #065F46; }
    .badge-warning { background: #FEF3C7; color: #92400E; }
    .badge-bad { background: #FEE2E2; color: #B91C1C; }

    .score-display {
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--text-main);
        line-height: 1;
        margin: 1rem 0;
    }
    .score-label {
        font-size: 0.9rem;
        color: var(--text-sub);
        font-weight: 500;
        text-transform: uppercase;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #F3F4F6;
    }
    .metric-row:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .metric-name { color: var(--text-sub); font-size: 0.95rem; }
    .metric-value { font-weight: 600; color: var(--text-main); }

    .recommendation-box {
        background: var(--primary-bg);
        border-left: 4px solid var(--primary-color);
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-top: 1.5rem;
    }
    .recommendation-title {
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        display: block;
    }
    .recommendation-text {
        color: var(--text-sub);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* --- EMPTY STATE --- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-sub);
    }
    .empty-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Hide default streamlit elements inside columns if needed */
    .element-container:has(.stImage) {
        text-align: center;
    }
    .stImage img {
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        max-height: 300px;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOGIC ỨNG DỤNG
# -----------------------------------------------------------------------------

def analyze_image(image):
    """Giả lập quá trình phân tích AI"""
    time.sleep(1.5) # Giả lập thời gian xử lý
    
    # Random kết quả để demo
    score = random.uniform(85, 99)
    if score > 90:
        status = "TƯƠI NGON"
        status_class = "badge-fresh"
        color = "#10B981"
        advice = "Thực phẩm ở trạng thái tốt nhất. Có thể sử dụng ngay hoặc bảo quản ngăn mát trong 2-3 ngày."
    elif score > 70:
        status = "CẦN LƯU Ý"
        status_class = "badge-warning"
        color = "#F59E0B"
        advice = "Thực phẩm bắt đầu có dấu hiệu giảm chất lượng. Nên chế biến ngay trong hôm nay, không nên lưu trữ thêm."
    else:
        status = "KHÔNG TƯƠI"
        status_class = "badge-bad"
        color = "#EF4444"
        advice = "Thực phẩm có dấu hiệu hư hỏng. Khuyến cáo không nên sử dụng để đảm bảo an toàn sức khỏe."
        
    return {
        "score": round(score, 1),
        "status": status,
        "class": status_class,
        "color": color,
        "advice": advice,
        "water": random.randint(60, 80),
        "protein": random.randint(15, 25),
        "date": "Hôm nay"
    }

# -----------------------------------------------------------------------------
# GIAO DIỆN CHÍNH (MAIN UI)
# -----------------------------------------------------------------------------

# 1. Header
col_header_l, col_header_r = st.columns([3, 1])
with col_header_l:
    st.markdown('<div class="brand-logo">🌿 <span>MonFresh</span></div>', unsafe_allow_html=True)
with col_header_r:
    st.markdown('<div class="header-status">Hệ thống hoạt động ổn định</div>', unsafe_allow_html=True)

st.markdown('<div style="height: 1px;"></div>', unsafe_allow_html=True) # Spacer

# 2. Main Content Grid
col_input, col_result = st.columns([1, 1.2], gap="large")

# --- CỘT TRÁI: INPUT ---
with col_input:
    st.markdown("""
    <div class="card">
        <div class="card-title">Nhập dữ liệu</div>
        <p style="color: var(--text-sub); margin-bottom: 1.5rem; font-size: 0.95rem;">
            Tải ảnh chụp thịt/heo hoặc sử dụng camera để hệ thống AI phân tích độ tươi ngay lập tức.
        </p>
    """, unsafe_allow_html=True)

    # Custom Tabs (Upload vs Camera)
    tab_upload, tab_camera = st.tabs(["📁 Tải ảnh lên", "📷 Chụp ảnh"])

    uploaded_file = None
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Chọn ảnh từ máy tính", 
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed",
            help="Kéo thả ảnh vào đây hoặc click để chọn"
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Ảnh đã tải lên", use_container_width=True)

    with tab_camera:
        cam_file = st.camera_input("Bật camera để chụp", label_visibility="collapsed")
        if cam_file:
            uploaded_file = cam_file
            st.image(cam_file, caption="Ảnh vừa chụp", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close card

    # Nút phân tích
    analyze_btn = st.button(
        "🔍 Bắt đầu phân tích ngay", 
        type="primary", 
        use_container_width=True,
        disabled=(uploaded_file is None)
    )

# --- CỘT PHẢI: KẾT QUẢ ---
with col_result:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Kết quả phân tích</div>', unsafe_allow_html=True)

    if uploaded_file and analyze_btn:
        with st.spinner("Đang xử lý hình ảnh..."):
            result = analyze_image(uploaded_file)
        
        # Hiển thị kết quả
        st.markdown(f'<div class="result-badge {result["class"]}">{result["status"]}</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f'<div class="score-label">Độ tươi</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-display" style="color: {result["color"]}">{result["score"]}%</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown("""
            <div style="margin-top: 10px;">
                <div class="metric-row">
                    <span class="metric-name">Độ ẩm ước tính</span>
                    <span class="metric-value">{water}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Cấu trúc protein</span>
                    <span class="metric-value">{prot}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Thời điểm kiểm tra</span>
                    <span class="metric-value">{date}</span>
                </div>
            </div>
            """.format(water=result['water'], prot=result['protein'], date=result['date']), unsafe_allow_html=True)

        st.markdown("""
        <div class="recommendation-box">
            <span class="recommendation-title">💡 Khuyến nghị từ MonFresh</span>
            <p class="recommendation-text">{advice}</p>
        </div>
        """.format(advice=result['advice']), unsafe_allow_html=True)

    else:
        # Empty State đẹp
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🥩</div>
            <h3 style="margin: 0; color: var(--text-main);">Chưa có kết quả</h3>
            <p style="margin: 10px 0 0 0; font-size: 0.95rem;">
                Vui lòng tải ảnh lên hoặc chụp ảnh ở cột bên trái<br>để bắt đầu quy trình kiểm tra độ tươi.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close card

# Footer nhỏ
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #9CA3AF; font-size: 0.85rem;">
    &copy; 2024 MonFresh Technology. Bảo vệ sức khỏe gia đình Việt.
</div>
""", unsafe_allow_html=True)
