import streamlit as st

# -----------------------------------------------------------------------------
# CẤU HÌNH TRANG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MonFresh - Kiểm Tra Độ Tươi Thực Phẩm",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary: #10B981;
        --primary-dark: #059669;
        --bg-color: #F9FAFB;
        --card-bg: #FFFFFF;
        --text-main: #1F2937;
        --text-muted: #6B7280;
        --border-color: #E5E7EB;
    }

    body {
        font-family: 'Inter', sans-serif;
        background: var(--bg-color);
        color: var(--text-main);
    }

    /* Ẩn elements thừa */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden; display: none;}
    
    .block-container {
        padding: 2rem 0;
        max-width: 900px;
    }

    /* Header */
    .header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .logo {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 0.5rem;
    }
    .tagline {
        color: var(--text-muted);
        font-size: 1rem;
    }

    /* Card chính */
    .main-card {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
    }

    /* File uploader */
    .stFileUploader {
        background: #F3F4F6;
        border: 2px dashed #D1D5DB;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s;
    }
    .stFileUploader:hover {
        border-color: var(--primary);
        background: #ECFDF5;
    }
    .stFileUploader label {
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Button */
    .stButton > button {
        background: var(--primary);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .stButton > button:disabled {
        background: #D1D5DB;
        transform: none;
        box-shadow: none;
    }

    /* Kết quả */
    .result-section {
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 2px solid #F3F4F6;
    }
    .result-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2rem;
        color: var(--text-muted);
    }
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }

    /* Info box */
    .info-box {
        background: #ECFDF5;
        border-left: 4px solid var(--primary);
        padding: 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-top: 1.5rem;
    }
    .info-title {
        font-weight: 600;
        color: var(--primary-dark);
        margin-bottom: 0.5rem;
    }
    .info-text {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Image preview */
    .stImage {
        text-align: center;
    }
    .stImage img {
        border-radius: 12px;
        max-height: 350px;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GIAO DIỆN CHÍNH
# -----------------------------------------------------------------------------

# Header
st.markdown("""
<div class="header">
    <div class="logo">🌿 MonFresh</div>
    <div class="tagline">Kiểm tra độ tươi thực phẩm bằng AI</div>
</div>
""", unsafe_allow_html=True)

# Main card
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader(
    "📤 Tải ảnh lên hoặc chụp ảnh",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed",
    help="Hỗ trợ định dạng JPG, JPEG, PNG"
)

if uploaded_file:
    st.image(uploaded_file, caption="Ảnh đã tải", use_container_width=True)

# Analyze button
col_btn, _ = st.columns([2, 1])
with col_btn:
    analyze_btn = st.button(
        "🔍 Phân tích ngay",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None)
    )

# Result section
if uploaded_file and analyze_btn:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">📊 Kết quả phân tích</div>', unsafe_allow_html=True)
    
    # Placeholder cho kết quả thực tế
    st.info("""
    **⚠️ Chưa tích hợp model AI**
    
    Để hiển thị kết quả thực tế, bạn cần tích hợp model machine learning vào phần này.
    
    Hiện tại ứng dụng đang ở chế độ demo giao diện.
    """)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-title">💡 Hướng dẫn tích hợp</div>
        <div class="info-text">
            Thêm code gọi model AI của bạn tại vị trí này để phân tích ảnh và trả về kết quả độ tươi thực tế.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🥩</div>
        <p>Tải ảnh lên và nhấn "Phân tích ngay" để xem kết quả</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #9CA3AF; font-size: 0.85rem;">
    &copy; 2024 MonFresh
</div>
""", unsafe_allow_html=True)
