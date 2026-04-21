import streamlit as st
import time
import random
from PIL import Image

# Cấu hình trang
st.set_page_config(
    page_title="MonFresh - Phân Tích Độ Tươi Thịt",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Tùy chỉnh giao diện cao cấp
st.markdown("""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --primary-color: #0F766E;
        --primary-light: #14B8A6;
        --primary-dark: #115E59;
        --accent-color: #F59E0B;
        --danger-color: #EF4444;
        --success-color: #10B981;
        --bg-color: #F8FAFC;
        --card-bg: #FFFFFF;
        --text-main: #1E293B;
        --text-muted: #64748B;
        --border-color: #E2E8F0;
        --radius-lg: 24px;
        --radius-md: 16px;
        --radius-sm: 8px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.025);
    }

    /* Global Styles */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-main);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--border-color);
        padding: 2rem 1rem;
    }
    
    .sidebar-logo {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-left: 1rem;
    }

    .nav-item {
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: var(--radius-md);
        color: var(--text-muted);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .nav-item.active {
        background-color: #F0FDFA;
        color: var(--primary-color);
        border: 1px solid #CCFBF1;
    }
    
    .nav-item:hover:not(.active) {
        background-color: #F1F5F9;
        color: var(--text-main);
    }

    /* Main Content Area */
    .main-content {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Header Section */
    .header-section {
        margin-bottom: 2.5rem;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-main);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Cards General */
    .custom-card {
        background: var(--card-bg);
        border-radius: var(--radius-lg);
        padding: 2rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    /* Upload Zone */
    .upload-zone {
        border: 2px dashed #CBD5E1;
        border-radius: var(--radius-lg);
        padding: 3rem 2rem;
        text-align: center;
        background-color: #F8FAFC;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .upload-zone:hover {
        border-color: var(--primary-light);
        background-color: #F0FDFA;
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .upload-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.5rem;
    }
    
    .upload-desc {
        color: var(--text-muted);
        font-size: 0.95rem;
    }

    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: var(--radius-md);
        background-color: #F1F5F9;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 1rem;
        border: none;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white !important;
        box-shadow: 0 4px 6px -1px rgba(15, 118, 110, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #E2E8F0;
        color: var(--text-main);
    }

    /* Result Components */
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .status-badge {
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-fresh {
        background-color: #DCFCE7;
        color: #166534;
    }
    
    .status-warning {
        background-color: #FEF3C7;
        color: #92400E;
    }
    
    .status-spoiled {
        background-color: #FEE2E2;
        color: #991B1B;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #F8FAFC;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
    }

    .recommendation-box {
        background: linear-gradient(135deg, #F0FDFA 0%, #ECFDF5 100%);
        border: 1px solid #A7F3D0;
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .rec-title {
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .rec-text {
        color: var(--text-main);
        line-height: 1.6;
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
        opacity: 0.5;
    }
    
    .empty-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.5rem;
    }

    /* Button Styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: var(--radius-sm);
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
        box-shadow: 0 4px 6px -1px rgba(15, 118, 110, 0.2);
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: 0 6px 8px -1px rgba(15, 118, 110, 0.3);
    }

    /* Hide file uploader default text */
    .stFileUploader p {
        display: none;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .metrics-grid {
            grid-template-columns: 1fr;
        }
        .main-content {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌿 MonFresh</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="nav-item active">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📷 Lịch sử quét</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">⚙️ Cài đặt</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("**Mẹo nhỏ:**\nChụp ảnh thịt dưới ánh sáng tự nhiên để có kết quả chính xác nhất.", icon="💡")

# --- MAIN CONTENT ---
def main():
    # Header
    st.markdown("""
    <div class="main-content">
        <div class="header-section">
            <h1 class="header-title">Phân tích độ tươi của thịt</h1>
            <p class="header-subtitle">Công nghệ AI giúp bạn nhận biết thực phẩm tươi sống nhanh chóng và chính xác.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        # Tabs for Input Method
        tab_upload, tab_camera = st.tabs(["📁 Tải ảnh lên", "📸 Chụp trực tiếp"])
        
        uploaded_file = None
        
        with tab_upload:
            st.markdown("""
            <div class="upload-zone">
                <span class="upload-icon">☁️</span>
                <div class="upload-title">Kéo thả ảnh vào đây</div>
                <div class="upload-desc">hoặc nhấn để chọn file (JPG, PNG)</div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
        with tab_camera:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <div class="upload-icon">📸</div>
                <div class="upload-title">Sử dụng Camera</div>
                <div class="upload-desc">Đặt thịt trên nền phẳng và chụp từ trên xuống</div>
            </div>
            """, unsafe_allow_html=True)
            camera_input = st.camera_input("Mở Camera", label_visibility="collapsed")
            if camera_input:
                uploaded_file = camera_input

        st.markdown('</div>', unsafe_allow_html=True) # End card

    with col_result:
        if uploaded_file is not None:
            # Display Result
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            # Top section: Image + Status
            c1, c2 = st.columns([1, 1])
            with c1:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with c2:
                st.markdown('<div class="result-header">', unsafe_allow_html=True)
                st.markdown('<div><h3 style="margin:0;">Kết quả phân tích</h3><p style="color:var(--text-muted); margin:5px 0 0;">Độ tin cậy: 98.5%</p></div>', unsafe_allow_html=True)
                
                # Simulate status based on random for demo
                status = random.choice(["fresh", "warning", "spoiled"])
                
                if status == "fresh":
                    badge_class = "status-fresh"
                    status_text = "TƯƠI NGON"
                    score = "95/100"
                    rec_title = "✅ Khuyến nghị:"
                    rec_content = "Thịt còn rất tươi. Bạn có thể bảo quản ngăn mát trong 2-3 ngày hoặc chế biến ngay để thưởng thức hương vị tốt nhất."
                    color_val = "#10B981"
                elif status == "warning":
                    badge_class = "status-warning"
                    status_text = "CẦN LƯU Ý"
                    score = "65/100"
                    rec_title = "⚠️ Khuyến nghị:"
                    rec_content = "Thịt bắt đầu có dấu hiệu giảm chất lượng. Nên chế biến ngay trong hôm nay, không nên bảo quản lâu hơn."
                    color_val = "#F59E0B"
                else:
                    badge_class = "status-spoiled"
                    status_text = "HƯ HỎNG"
                    score = "20/100"
                    rec_title = "❌ Cảnh báo:"
                    rec_content = "Thịt đã hỏng, không nên sử dụng. Vui lòng bỏ bỏ để đảm bảo an toàn sức khỏe."
                    color_val = "#EF4444"
                
                st.markdown(f'<div class="status-badge {badge_class}">{status_text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True) # End header
                
                # Metrics
                st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" style="color:{color_val}">{score}</div>
                        <div class="metric-label">Điểm chất lượng</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">24h</div>
                        <div class="metric-label">Thời gian ước tính</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">A+</div>
                        <div class="metric-label">Hạng</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendation
                st.markdown(f"""
                <div class="recommendation-box">
                    <div class="rec-title">{rec_title}</div>
                    <div class="rec-text">{rec_content}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True) # End card
            
        else:
            # Empty State
            st.markdown("""
            <div class="custom-card" style="display:flex; align-items:center; justify-content:center;">
                <div class="empty-state">
                    <div class="empty-icon">🥩</div>
                    <div class="empty-title">Chưa có dữ liệu phân tích</div>
                    <p style="max-width: 400px; line-height: 1.6;">Vui lòng tải lên một bức ảnh thịt hoặc sử dụng camera để hệ thống AI tiến hành kiểm tra độ tươi ngay lập tức.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
