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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #10B981;
        --primary-dark: #059669;
        --primary-light: #D1FAE5;
        --bg-color: #F0FDF4;
        --card-bg: #FFFFFF;
        --text-main: #1F2937;
        --text-muted: #6B7280;
        --border-color: #E5E7EB;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.15);
    }

    * {
        box-sizing: border-box;
    }

    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
        color: var(--text-main);
        min-height: 100vh;
    }

    /* Ẩn elements thừa */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden; display: none;}
    
    .block-container {
        padding: 3rem 0;
        max-width: 960px;
    }

    /* Header */
    .header {
        text-align: center;
        margin-bottom: 3.5rem;
        padding: 2rem 0;
    }
    .logo {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-dark), var(--primary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.75rem;
        letter-spacing: -0.5px;
    }
    .tagline {
        color: var(--text-muted);
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Main card */
    .main-card {
        background: var(--card-bg);
        border-radius: 24px;
        padding: 3rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(255,255,255,0.8);
        margin-bottom: 2rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* File uploader */
    .stFileUploader {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        border: 2px dashed #D1D5DB;
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    .stFileUploader:hover {
        border-color: var(--primary);
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        transform: translateY(-2px);
    }
    .stFileUploader label {
        color: var(--text-muted);
        font-weight: 500;
        font-size: 1rem;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 1rem 2.5rem;
        border-radius: 14px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
    }
    .stButton > button:disabled {
        background: #D1D5DB;
        transform: none;
        box-shadow: none;
        cursor: not-allowed;
    }

    /* Result section */
    .result-section {
        margin-top: 2.5rem;
        padding-top: 2.5rem;
        border-top: 2px solid #F3F4F6;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--text-muted);
    }
    .empty-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        opacity: 0.5;
    }
    .empty-text {
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 5px solid var(--primary);
        padding: 1.75rem;
        border-radius: 0 16px 16px 0;
        margin-top: 2rem;
        box-shadow: var(--shadow-sm);
    }
    .info-title {
        font-weight: 600;
        color: var(--primary-dark);
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    .info-text {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Features section */
    .features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    .feature-item {
        background: #F9FAFB;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        transition: all 0.3s;
    }
    .feature-item:hover {
        background: var(--primary-light);
        transform: translateY(-3px);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }
    .feature-title {
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Image preview */
    .stImage {
        text-align: center;
        margin: 2rem 0;
    }
    .stImage img {
        border-radius: 16px;
        max-height: 400px;
        object-fit: contain;
        box-shadow: var(--shadow-md);
    }

    /* Spacer */
    .spacer {
        height: 1.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #9CA3AF;
        font-size: 0.9rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 3rem;
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
    <div class="tagline">Giải pháp AI thông minh giúp bạn kiểm tra độ tươi của thực phẩm chỉ với một bức ảnh</div>
</div>
""", unsafe_allow_html=True)

# Features section
st.markdown("""
<div class="features">
    <div class="feature-item">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Nhanh Chóng</div>
        <div class="feature-desc">Kết quả phân tích trong vài giây</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Chính Xác</div>
        <div class="feature-desc">AI được huấn luyện trên hàng ngàn mẫu</div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">🛡️</div>
        <div class="feature-title">An Toàn</div>
        <div class="feature-desc">Bảo mật tuyệt đối hình ảnh của bạn</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# Main card
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="section-title">📤 Bước 1: Tải ảnh lên</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Chọn ảnh thực phẩm cần kiểm tra (JPG, JPEG, PNG)",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed",
    help="Bạn có thể tải ảnh từ thư viện hoặc chụp trực tiếp"
)

if uploaded_file:
    st.image(uploaded_file, caption="Ảnh đã tải lên", use_container_width=True)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# Analyze button
st.markdown('<div class="section-title">🔍 Bước 2: Phân tích</div>', unsafe_allow_html=True)
analyze_btn = st.button(
    "✨ Phân Tích Ngay",
    type="primary",
    use_container_width=True,
    disabled=(uploaded_file is None)
)

# Result section
if uploaded_file and analyze_btn:
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Kết Quả Phân Tích</div>', unsafe_allow_html=True)
    
    # Placeholder cho kết quả thực tế
    st.warning("""
    **🤖 Chưa tích hợp model AI**
    
    Ứng dụng đang ở chế độ demo giao diện. Để hiển thị kết quả thực tế, 
    bạn cần tích hợp model machine learning vào phần này.
    
    **Hướng dẫn phát triển:**
    - Thêm code gọi model AI tại vị trí này
    - Xử lý ảnh đầu vào qua model
    - Hiển thị kết quả độ tươi và khuyến nghị
    """)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-title">💡 Gợi ý tích hợp</div>
        <div class="info-text">
            Sử dụng các framework như TensorFlow, PyTorch hoặc các API AI có sẵn 
            để phân tích hình ảnh. Kết quả nên bao gồm: trạng thái độ tươi, 
            ước tính thời gian bảo quản còn lại, và khuyến nghị sử dụng.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🥩</div>
        <div class="empty-text">
            <strong>Tải ảnh lên và nhấn "Phân Tích Ngay"</strong><br>
            để nhận kết quả kiểm tra độ tươi thực phẩm
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    &copy; 2024 MonFresh - Giải pháp AI kiểm tra độ tươi thực phẩm<br>
    Phát triển với ❤️ bởi đội ngũ MonFresh
</div>
""", unsafe_allow_html=True)
