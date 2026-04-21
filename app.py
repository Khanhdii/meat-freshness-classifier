import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================================================
# CẤU HÌNH TRANG & CUSTOM CSS - PROFESSIONAL DESIGN (MATCHING INDEX.HTML)
# ============================================================================

st.set_page_config(
    page_title="MonFresh - AI Meat Freshness Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean minimal design with theme colors
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Theme Color Variables - Nature & Environment */
    :root {
        --color-primary: #2E7D32;      /* Forest Green - Chủ đạo */
        --color-primary-dark: #1B5E20; /* Dark Forest */
        --color-primary-light: #4CAF50; /* Fresh Leaf */
        --color-accent: #81C784;       /* Soft Green */
        --color-bg: #F1F8E9;           /* Very light green background */
        --color-card: #FFFFFF;         /* Pure white cards */
        --color-text: #1B3320;         /* Very dark green text */
        --color-text-muted: #556B58;   /* Muted green-gray */
        --color-border: #C8E6C9;       /* Light border */
        --color-fresh: #2E7D32;        /* Fresh status */
        --color-half: #F9A825;         /* Half-fresh status */
        --color-spoiled: #C62828;      /* Spoiled status */
    }
    
    /* Base styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--color-bg);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header - Simple clean design */
    .header-container {
        background: var(--color-card);
        padding: 1.5rem 2rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
        border-bottom: 2px solid var(--color-border);
    }
    
    .header-content {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--color-primary-dark);
        margin-bottom: 0.25rem;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        color: var(--color-text-muted);
        font-weight: 400;
    }
    
    /* Stats bar - Clean horizontal layout */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
        background: var(--color-card);
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid var(--color-border);
    }
    
    .stat-item {
        text-align: center;
        padding: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--color-primary);
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Card styling - Minimal clean */
    .card {
        background: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border);
    }
    
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--color-primary);
    }
    
    /* Result display - Clean minimal */
    .result-display {
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        background: var(--color-card);
        border: 2px solid;
    }
    
    .result-display.fresh {
        border-color: var(--color-fresh);
        background: var(--color-bg);
    }
    
    .result-display.half {
        border-color: var(--color-half);
        background: #FFF8E1;
    }
    
    .result-display.spoiled {
        border-color: var(--color-spoiled);
        background: #FFEBEE;
    }
    
    .result-status {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-status.fresh { color: var(--color-fresh); }
    .result-status.half { color: var(--color-half); }
    .result-status.spoiled { color: var(--color-spoiled); }
    
    .result-subtitle {
        font-size: 0.875rem;
        color: var(--color-text-muted);
        margin-bottom: 1rem;
    }
    
    .result-confidence {
        font-size: 0.875rem;
        color: var(--color-primary);
        font-weight: 500;
    }
    
    /* Probability bars - Clean design */
    .probability-item {
        margin-bottom: 1rem;
    }
    
    .probability-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    
    .probability-label {
        font-weight: 500;
        color: var(--color-primary);
    }
    
    .probability-value {
        color: var(--color-text-muted);
    }
    
    .probability-bar {
        height: 6px;
        background: var(--color-border);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .probability-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    .probability-fill.fresh { background: var(--color-fresh); }
    .probability-fill.half { background: var(--color-half); }
    .probability-fill.spoiled { background: var(--color-spoiled); }
    
    /* Recommendation boxes - Simple clean */
    .recommendation {
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.875rem;
        line-height: 1.7;
        border-left: 3px solid;
    }
    
    .recommendation.fresh {
        background: var(--color-bg);
        border-left-color: var(--color-fresh);
        color: var(--color-primary-dark);
    }
    
    .recommendation.half {
        background: #FFF8E1;
        border-left-color: var(--color-half);
        color: #B7791F;
    }
    
    .recommendation.spoiled {
        background: #FFEBEE;
        border-left-color: var(--color-spoiled);
        color: #C62828;
    }
    
    /* Sidebar - Minimal design */
    [data-testid="stSidebar"] {
        background: var(--color-bg);
        border-right: 1px solid var(--color-border);
    }
    
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--color-primary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--color-border);
    }
    
    .class-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: var(--color-card);
        border-radius: 6px;
        margin-bottom: 0.5rem;
        border: 1px solid var(--color-border);
    }
    
    .class-color {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        flex-shrink: 0;
    }
    
    .class-color.fresh { background: var(--color-fresh); }
    .class-color.half { background: var(--color-half); }
    .class-color.spoiled { background: var(--color-spoiled); }
    
    .class-name {
        font-weight: 500;
        color: var(--color-primary);
        font-size: 0.875rem;
    }
    
    .class-desc {
        font-size: 0.75rem;
        color: var(--color-text-muted);
    }
    
    /* Empty state - Minimal */
    .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        color: var(--color-text-muted);
        background: var(--color-bg);
        border: 1px solid var(--color-border);
        border-radius: 8px;
    }
    
    .empty-state-title {
        font-weight: 600;
        color: var(--color-primary);
        margin-bottom: 0.5rem;
        font-size: 0.9375rem;
    }
    
    .empty-state-desc {
        font-size: 0.8125rem;
        color: var(--color-text-muted);
    }
    
    /* Tips section - Clean */
    .tips-section {
        background: var(--color-bg);
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        border: 1px solid var(--color-border);
    }
    
    .tips-title {
        font-weight: 600;
        color: var(--color-primary);
        margin-bottom: 0.5rem;
        font-size: 0.8125rem;
    }
    
    .tips-list {
        margin: 0;
        padding-left: 1.25rem;
        color: var(--color-text-muted);
        font-size: 0.8125rem;
        line-height: 1.75;
    }
    
    .tips-list li {
        margin-bottom: 0.25rem;
    }
    
    /* Note boxes - Simple */
    .note-box {
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.8125rem;
        border: 1px solid;
    }
    
    .note-box.info { 
        background: #E3F2FD; 
        border-color: #BBDEFB; 
        color: #1565C0; 
    }
    
    .note-box.warning { 
        background: #FFF8E1; 
        border-color: #FFE082; 
        color: #F9A825; 
    }
    
    /* File uploader - Minimal */
    .stFileUploader > div {
        border: 1px solid var(--color-border);
        border-radius: 6px;
        padding: 1rem;
        background: var(--color-card);
    }
    
    /* Button styling - Clean */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.625rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button[type="primary"] {
        background: var(--color-primary);
        color: white;
        border: none;
    }
    
    .stButton > button[type="primary"]:hover {
        background: var(--color-primary-dark);
    }
    
    /* Tabs - Minimal block style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--color-border);
        padding: 0.25rem;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: var(--color-text-muted);
        font-size: 0.875rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--color-card);
        color: var(--color-primary);
    }
    
    /* Image wrapper - Clean */
    .image-wrapper {
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid var(--color-border);
        background: var(--color-card);
    }
    
    /* Layout utilities */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Footer - Simple */
    .footer {
        background: var(--color-card);
        border-top: 1px solid var(--color-border);
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        color: var(--color-text-muted);
        font-size: 0.8125rem;
    }
    
    /* Scrollbar - Minimal */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--color-border);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--color-accent);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--color-primary);
    }
</style>
""", unsafe_allow_html=True)

# Constants từ model training
INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES = {0: 'FRESH', 1: 'HALF', 2: 'SPOILED'}
CLASS_NAMES_VI = {0: 'Tươi', 1: 'Bán tươi', 2: 'Hỏng'}
CLASS_COLORS = {0: '#22c55e', 1: '#f59e0b', 2: '#ef4444'}

@st.cache_resource
def load_model():
    """Load model đã được huấn luyện"""
    try:
        model = tf.keras.models.load_model('dw_sppf_net.keras')
        return model
    except:
        try:
            model = tf.keras.models.load_model('dw_sppf_net.keras')
            return model
        except Exception as e:
            st.error(f"Không thể load model: {e}")
            return None

def preprocess_image(image):
    """Tiền xử lý ảnh theo cùng cách như khi training"""
    # Resize ảnh về kích thước 224x224
    image = image.resize((224, 224))
    
    # Chuyển thành numpy array
    img_array = np.array(image)
    
    # Đảm bảo có 3 channels (RGB)
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    elif len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Chuẩn hóa pixel values về [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Thêm batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image):
    """Dự đoán độ tươi của thịt"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Lấy class có xác suất cao nhất
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence, predictions[0]

def analyze_image(model, image, result_col):
    """Phân tích ảnh và hiển thị kết quả theo phong cách tối giản"""
    with st.spinner(""):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with result_col:
                class_name = CLASS_NAMES[predicted_class]
                class_name_vi = CLASS_NAMES_VI[predicted_class]
                result_type = ['fresh', 'half', 'spoiled'][predicted_class]
                
                # Result display chính
                st.markdown(f"""
                <div class="result-display {result_type}">
                    <div class="result-status {result_type}">{class_name_vi}</div>
                    <div class="result-subtitle">{class_name}</div>
                    <div class="result-confidence">Độ tin cậy: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Biểu đồ xác suất
                st.markdown('<div class="card" style="margin-top: 1.5rem;"><div class="card-header"><div class="card-title">Xác suất phân loại</div></div>', unsafe_allow_html=True)
                
                for class_id, prob in zip(CLASS_NAMES.keys(), all_predictions):
                    cn = CLASS_NAMES[class_id]
                    cn_vi = CLASS_NAMES_VI[class_id]
                    fill_type = ['fresh', 'half', 'spoiled'][class_id]
                    
                    st.markdown(f"""
                    <div class="probability-item">
                        <div class="probability-header">
                            <span class="probability-label">{cn_vi}</span>
                            <span class="probability-value">{prob:.1%}</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill {fill_type}" style="width: {prob*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Khuyến nghị
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Khuyến nghị</div></div>', unsafe_allow_html=True)
                
                if predicted_class == 0:
                    st.markdown("""
                    <div class="recommendation fresh">
                        <strong>Thịt còn tươi</strong><br>
                        Sản phẩm ở trạng thái tốt nhất, có thể sử dụng an toàn ngay lập tức.
                        Nên bảo quản ở nhiệt độ thích hợp để duy trì độ tươi.
                    </div>
                    """, unsafe_allow_html=True)
                elif predicted_class == 1:
                    st.markdown("""
                    <div class="recommendation half">
                        <strong>Thịt bán tươi</strong><br>
                        Sản phẩm vẫn có thể sử dụng nhưng nên chế biến sớm.
                        Kiểm tra kỹ mùi và kết cấu trước khi sử dụng.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="recommendation spoiled">
                        <strong>Thịt đã hỏng</strong><br>
                        Không nên sử dụng sản phẩm này.
                        Có nguy cơ gây ngộ độc thực phẩm.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

def main():
    # Header đơn giản
    st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <div class="header-title">MonFresh</div>
            <div class="header-subtitle">Phân tích độ tươi của thịt bằng AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Không thể load model. Vui lòng kiểm tra file model.")
        return
    
    # Stats bar
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-value">{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}</div>
            <div class="stat-label">Độ phân giải</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{len(CLASS_NAMES)}</div>
            <div class="stat-label">Lớp phân loại</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">AI</div>
            <div class="stat-label">Công nghệ</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">&lt; 1s</div>
            <div class="stat-label">Xử lý</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar thông tin
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">Các lớp phân loại</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-color fresh"></div>
            <div>
                <div class="class-name">Tươi</div>
                <div class="class-desc">Sản phẩm chất lượng tốt</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-color half"></div>
            <div>
                <div class="class-name">Bán tươi</div>
                <div class="class-desc">Cần sử dụng sớm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-color spoiled"></div>
            <div>
                <div class="class-name">Hỏng</div>
                <div class="class-desc">Không nên sử dụng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section" style="margin-top: 1.5rem;">
            <div class="sidebar-title">Hướng dẫn</div>
        </div>
        <div style="font-size: 0.8125rem; line-height: 1.7; color: var(--color-text-muted);">
            <strong>Upload:</strong> Chọn ảnh từ thiết bị<br>
            <strong>Camera:</strong> Chụp ảnh trực tiếp<br>
            <strong>Kết quả:</strong> Xem ở cột bên phải
        </div>
        """, unsafe_allow_html=True)
    
    # Main content container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Main content - Tabs cho Upload và Camera
    tab1, tab2 = st.tabs(["Upload ảnh", "Chụp ảnh từ Camera"])
    
    # Tab 1: Upload ảnh
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card"><div class="card-header"><div class="card-title">Tải ảnh lên</div></div>', unsafe_allow_html=True)
            
            # Custom upload zone styling
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-icon"></div>
                <div class="upload-title">Click để tải ảnh lên hoặc kéo thả</div>
                <div class="upload-desc">Hỗ trợ PNG, JPG, JPEG (tối đa 800x400px)
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "",
                type=['png', 'jpg', 'jpeg'],
                help="Hỗ trợ định dạng PNG, JPG, JPEG",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
                st.image(image, caption="", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                if st.button("Phân tích độ tươi", type="primary", key="upload_predict", use_container_width=True):
                    analyze_image(model, image, col2)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if uploaded_file is None:
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Kết quả phân loại</div></div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="empty-state">
                    
                    <div class="empty-state-title">Chưa có kết quả</div>
                    <div class="empty-state-desc">Vui lòng tải ảnh lên ở cột bên trái để bắt đầu phân tích</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tips-section">
                    <div class="tips-title">Mẹo để có kết quả tốt nhất</div>
                    <ul class="tips-list">
                        <li>Sử dụng ảnh có độ phân giải cao</li>
                        <li>Đảm bảo ánh sáng đủ và đều</li>
                        <li>Thịt nên được chụp rõ nét, chiếm phần lớn khung hình</li>
                        <li>Tránh bóng đổ che khuất bề mặt thịt</li>
                        <li>Nên chụp từ góc nhìn trực diện</li>
                    </ul>
                </div>
                
                <div class="note-box info">
                    <strong>Lưu ý:</strong> Kết quả phân tích mang tính chất tham khảo. 
                    Luôn kiểm tra thêm bằng các giác quan (mùi, màu sắc, kết cấu) trước khi sử dụng.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Chụp ảnh từ camera
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card"><div class="card-header"><div class="card-title">Chụp ảnh trực tiếp</div></div>', unsafe_allow_html=True)
            
            if 'camera_enabled' not in st.session_state:
                st.session_state.camera_enabled = False
            
            if not st.session_state.camera_enabled:
                st.markdown("""
                <div class="empty-state">
                    
                    <div class="empty-state-title">Camera đang tắt</div>
                    <div class="empty-state-desc">Click nút "Bật Camera" để mở camera và chụp ảnh</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Bật Camera", type="primary", key="enable_camera", use_container_width=True):
                    st.session_state.camera_enabled = True
                    st.rerun()
            else:
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if st.button("Tắt Camera", key="disable_camera", use_container_width=True):
                        st.session_state.camera_enabled = False
                        st.rerun()
                with col_b:
                    if st.button("Làm mới", key="new_photo", use_container_width=True):
                        pass
                
                camera_photo = st.camera_input(
                    "", 
                    help="Click vào nút camera để chụp ảnh",
                    key="camera_input",
                    label_visibility="collapsed"
                )
                
                if camera_photo is not None:
                    camera_image = Image.open(camera_photo)
                    st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
                    st.image(camera_image, caption="", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                    if st.button("Phân tích độ tươi", type="primary", key="camera_predict", use_container_width=True):
                        analyze_image(model, camera_image, col2)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if not st.session_state.camera_enabled:
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Kết quả phân loại</div></div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="empty-state">
                    
                    <div class="empty-state-title">Chưa có kết quả</div>
                    <div class="empty-state-desc">Vui lòng bật camera ở cột bên trái để bắt đầu</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tips-section">
                    <div class="tips-title">Hướng dẫn chụp ảnh</div>
                    <ul class="tips-list">
                        <li><strong>Đặt thịt</strong> trên nền sáng, phẳng</li>
                        <li><strong>Giữ camera ổn định</strong> khi chụp</li>
                        <li><strong>Đảm bảo ánh sáng</strong> đủ sáng và đều</li>
                        <li><strong>Chụp từ góc nhìn trực diện</strong></li>
                        <li><strong>Tránh phản quang</strong> và bóng đổ</li>
                    </ul>
                </div>
                
                <div class="note-box warning">
                    <strong>Lưu ý:</strong> Camera chỉ bật khi cần để tiết kiệm tài nguyên hệ thống.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif camera_photo is None:
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Kết quả phân loại</div></div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="empty-state">
                    
                    <div class="empty-state-title">Sẵn sàng chụp</div>
                    <div class="empty-state-desc">Vui lòng chụp ảnh ở cột bên trái để bắt đầu phân tích</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tips-section">
                    <div class="tips-title">Camera đã sẵn sàng!</div>
                    <ul class="tips-list">
                        <li>Click vào nút camera để chụp ảnh</li>
                        <li>Có thể "Tắt Camera" khi không dùng</li>
                    </ul>
                </div>
                
                <div class="note-box success">
                    <strong>Sẵn sàng phân tích!</strong><br>
                    Chất lượng ảnh tốt sẽ cho kết quả chính xác hơn.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        MonFresh © 2024 - Phân tích độ tươi của thịt bằng AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
