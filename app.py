import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================================================
# CẤU HÌNH TRANG & CUSTOM CSS - FLAT MODERN DESIGN
# ============================================================================

st.set_page_config(
    page_title="MeatFresh AI - Phân loại độ tươi thịt",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Flat Modern Design (No Shadows)
CUSTOM_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset */
* {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

/* Main App Background */
.stApp {
    background: #ffffff;
}

/* Header - Clean Flat Design */
.header-container {
    background: #0f172a;
    padding: 1.25rem 0;
    border-bottom: 1px solid #e2e8f0;
}

.header-content {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-logo {
    display: flex;
    align-items: center;
    gap: 0.875rem;
}

.logo-icon {
    width: 44px;
    height: 44px;
    background: #3b82f6;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: white;
}

.header-title {
    color: #0f172a;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.025em;
}

.header-subtitle {
    color: #64748b;
    font-size: 0.875rem;
    font-weight: 400;
}

/* Stats Bar - Flat Cards */
.stats-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1.5rem 2rem;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
}

.stat-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem;
    text-align: center;
}

.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.25rem;
}

.stat-label {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Main Content */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

/* Cards - Flat Design */
.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #f1f5f9;
}

.card-title {
    color: #0f172a;
    font-size: 1.125rem;
    font-weight: 600;
}

.card-badge {
    background: #0f172a;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Result Box - Flat with Border */
.result-box {
    background: #f8fafc;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    border: 2px solid #e2e8f0;
}

.result-box.success {
    background: #f0fdf4;
    border-color: #22c55e;
}

.result-box.warning {
    background: #fffbeb;
    border-color: #f59e0b;
}

.result-box.error {
    background: #fef2f2;
    border-color: #ef4444;
}

.result-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #0f172a;
}

.result-confidence {
    font-size: 0.875rem;
    color: #64748b;
    font-weight: 500;
    background: white;
    padding: 0.375rem 1rem;
    border-radius: 9999px;
    display: inline-block;
    margin-top: 0.75rem;
    border: 1px solid #e2e8f0;
}

/* Probability Bars - Flat */
.probability-item {
    margin-bottom: 1rem;
}

.probability-bar-bg {
    background: #e2e8f0;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}

.probability-bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* Info Boxes - Flat */
.info-box, .success-box, .warning-box, .error-box {
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    border: 1px solid;
}

.info-box {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1e40af;
}

.success-box {
    background: #f0fdf4;
    border-color: #bbf7d0;
    color: #166534;
}

.warning-box {
    background: #fffbeb;
    border-color: #fef3c7;
    color: #92400e;
}

.error-box {
    background: #fef2f2;
    border-color: #fee2e2;
    color: #991b1b;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background: #0f172a;
    border-radius: 4px;
}

/* Buttons - Flat */
.stButton > button {
    background: #0f172a;
    color: white;
    border: none;
    padding: 0.625rem 1.5rem;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.875rem;
}

.stButton > button:hover {
    background: #1e293b;
}

/* Sidebar - Clean */
[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #0f172a !important;
}

.sidebar-section {
    margin: 1.5rem 0;
}

.sidebar-title {
    color: #0f172a;
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.class-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin: 0.5rem 0;
}

.class-indicator {
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 1rem;
    color: white;
}

.class-info-title {
    font-weight: 600;
    color: #0f172a;
    font-size: 0.875rem;
}

.class-info-desc {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.125rem;
}

/* Tabs - Flat Underline */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    justify-content: center;
    background: transparent;
    border-bottom: 1px solid #e2e8f0;
    padding: 0;
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.75rem 2rem;
    font-weight: 500;
    color: #64748b;
    border: none;
    background: transparent;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #0f172a;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    background: transparent;
    color: #0f172a;
    border-bottom-color: #0f172a;
}

/* File Uploader - Flat */
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1;
    border-radius: 8px;
    padding: 2rem;
    background: #f8fafc;
}

[data-testid="stFileUploader"]:hover {
    border-color: #0f172a;
    background: #f1f5f9;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 2.5rem;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px dashed #cbd5e1;
}

.empty-state-icon {
    font-size: 2.5rem;
    color: #cbd5e1;
    margin-bottom: 0.75rem;
}

.empty-state-title {
    font-size: 1rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 0.25rem;
}

.empty-state-desc {
    color: #64748b;
    font-size: 0.875rem;
}

/* Tips Section */
.tips-section {
    background: #f0f9ff;
    padding: 1.25rem;
    border-radius: 6px;
    margin-top: 1rem;
    border: 1px solid #bae6fd;
}

.tips-title {
    font-weight: 600;
    color: #0369a1;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
}

.tips-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.tips-list li {
    padding: 0.375rem 0;
    color: #0c4a6e;
    font-size: 0.875rem;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}

.tips-list li:before {
    content: "•";
    color: #0ea5e9;
    font-weight: bold;
}

/* Footer */
.footer {
    background: #f8fafc;
    color: #64748b;
    padding: 2rem 0;
    margin-top: 3rem;
    text-align: center;
    border-top: 1px solid #e2e8f0;
}

.footer-content {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 2rem;
}

.footer-brand {
    font-size: 1.25rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 0.5rem;
}

.footer-text {
    font-size: 0.875rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.footer-disclaimer {
    font-size: 0.75rem;
    color: #94a3b8;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Layout Fixes */
.block-container {
    padding-top: 0;
}

div[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

/* Image Styling */
.stImage img {
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Divider */
hr {
    border-color: #e2e8f0;
    margin: 1.5rem 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Constants từ model training
INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES = {0: 'FRESH', 1: 'HALF', 2: 'SPOILED'}
CLASS_NAMES_VI = {0: 'Tươi', 1: 'Bán tươi', 2: 'Hỏng'}
CLASS_COLORS = {0: '#28a745', 1: '#ffc107', 2: '#dc3545'}

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
    """Phân tích ảnh và hiển thị kết quả theo phong cách thương mại"""
    with st.spinner(""):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with result_col:
                class_name = CLASS_NAMES[predicted_class]
                class_name_vi = CLASS_NAMES_VI[predicted_class]
                color_class = ['success', 'warning', 'error'][predicted_class]
                
                # Result box với styling đẹp - không dùng icon
                st.markdown(f"""
                <div class="result-box {color_class}">
                    <div class="result-title" style="color: {CLASS_COLORS[predicted_class]}">{class_name_vi}</div>
                    <div style="font-size: 1.2rem; margin-bottom: 0.5rem; color: #666;">{class_name}</div>
                    <div class="result-confidence">Độ tin cậy: {confidence:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Biểu đồ chi tiết các xác suất
                st.markdown('<div class="card" style="margin-top: 1.5rem;"><div class="card-title">Chi tiết xác suất</div>', unsafe_allow_html=True)
                
                for i, (class_id, prob) in enumerate(zip(CLASS_NAMES.keys(), all_predictions)):
                    cn = CLASS_NAMES[class_id]
                    cn_vi = CLASS_NAMES_VI[class_id]
                    bar_color = CLASS_COLORS[class_id]
                    
                    # Custom progress bar với màu sắc - không dùng icon
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.375rem;">
                            <span style="font-weight: 600; color: #333;">{cn_vi} ({cn})</span>
                            <span style="color: #666;">{prob:.2%}</span>
                        </div>
                        <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: {bar_color}; width: {prob*100}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Khuyến nghị
                st.markdown('<div class="card"><div class="card-title">Khuyến nghị</div>', unsafe_allow_html=True)
                
                if predicted_class == 0:
                    st.markdown("""
                    <div class="success-box">
                        <strong style="color: #28a745;">Thịt còn tươi</strong><br>
                        Sản phẩm ở trạng thái tốt nhất, có thể sử dụng an toàn ngay lập tức.
                        Nên bảo quản ở nhiệt độ thích hợp để duy trì độ tươi.
                    </div>
                    """, unsafe_allow_html=True)
                elif predicted_class == 1:
                    st.markdown("""
                    <div class="warning-box">
                        <strong style="color: #ffc107;">Thịt bán tươi</strong><br>
                        Sản phẩm vẫn có thể sử dụng nhưng nên chế biến sớm.
                        Kiểm tra kỹ mùi và kết cấu trước khi sử dụng.
                        Không nên bảo quản lâu thêm.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-box">
                        <strong style="color: #dc3545;">Thịt đã hỏng</strong><br>
                        <strong>Không nên sử dụng sản phẩm này.</strong>
                        Có nguy cơ gây ngộ độc thực phẩm và ảnh hưởng đến sức khỏe.
                        Vui lòng loại bỏ sản phẩm đúng cách.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

def main():
    # Header chuyên nghiệp - không dùng emoji
    st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <div class="header-title">MeatFresh AI</div>
            <div class="header-subtitle">Hệ thống phân loại độ tươi thịt thông minh sử dụng AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Không thể load model. Vui lòng kiểm tra file model.")
        return
    
    # Stats bar - Professional layout
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}</div>
            <div class="stat-label">Độ phân giải</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(CLASS_NAMES)}</div>
            <div class="stat-label">Lớp phân loại</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">DL</div>
            <div class="stat-label">Công nghệ AI</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">&lt; 1s</div>
            <div class="stat-label">Thời gian xử lý</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar thông tin - Professional styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div class="sidebar-title">Thông tin hệ thống</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box" style="background: rgba(255,255,255,0.1); border-left: 4px solid white;">
            <strong style="color: white;">Kích thước đầu vào:</strong><br>
            <span style="color: rgba(255,255,255,0.9);">{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]} pixels</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">Các lớp phân loại</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-indicator" style="background: #22c55e;">✓</div>
            <div>
                <div class="class-info-title">Tươi (Fresh)</div>
                <div class="class-info-desc">Sản phẩm chất lượng tốt nhất</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-indicator" style="background: #f59e0b;">!</div>
            <div>
                <div class="class-info-title">Bán tươi (Half)</div>
                <div class="class-info-desc">Cần sử dụng sớm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-indicator" style="background: #ef4444;">✕</div>
            <div>
                <div class="class-info-title">Hỏng (Spoiled)</div>
                <div class="class-info-desc">Không nên sử dụng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">Hướng dẫn</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="font-size: 0.9rem; line-height: 1.8; color: rgba(255,255,255,0.9);">
            <strong>Upload:</strong><br>
            • Chọn ảnh từ thiết bị<br>
            • Click "Phân tích độ tươi"<br>
            • Xem kết quả chi tiết<br><br>
            
            <strong>Camera:</strong><br>
            • Bật camera để kích hoạt<br>
            • Chụp ảnh thịt cần phân loại<br>
            • Nhận kết quả ngay lập tức<br><br>
            
            <em>Camera chỉ bật khi cần để tiết kiệm tài nguyên</em>
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
            uploaded_file = st.file_uploader(
                "Chọn ảnh thịt cần phân loại",
                type=['png', 'jpg', 'jpeg'],
                help="Hỗ trợ định dạng PNG, JPG, JPEG"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
                
                if st.button("Phân tích độ tươi", type="primary", key="upload_predict"):
                    analyze_image(model, image, col2)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if uploaded_file is None:
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Kết quả phân loại</div></div>', unsafe_allow_html=True)
                st.info("Vui lòng tải ảnh lên ở cột bên trái để bắt đầu phân tích")
                
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
                
                <div style="background: #f0f9ff; padding: 1.25rem; border-radius: 6px; margin-top: 1rem; border: 1px solid #bae6fd;">
                    <strong style="color: #0369a1;">Lưu ý:</strong> Kết quả phân tích mang tính chất tham khảo. 
                    Luôn kiểm tra thêm bằng các giác quan (mùi, màu sắc, kết cấu) trước khi sử dụng.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Chụp ảnh từ camera
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card"><div class="card-header"><div class="card-title">Chụp ảnh trực tiếp</div></div>', unsafe_allow_html=True)
            
            if 'camera_enabled' not in st.session_state:
                st.session_state.camera_enabled = False
            
            if not st.session_state.camera_enabled:
                if st.button("Bật Camera", type="primary", key="enable_camera"):
                    st.session_state.camera_enabled = True
                    st.rerun()
                
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">📷</div>
                    <div class="empty-state-title">Camera đang tắt</div>
                    <div class="empty-state-desc">Click nút "Bật Camera" để mở camera và chụp ảnh</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if st.button("Tắt Camera", key="disable_camera"):
                        st.session_state.camera_enabled = False
                        st.rerun()
                with col_b:
                    if st.button("Làm mới", key="new_photo"):
                        pass
                
                camera_photo = st.camera_input(
                    "Chụp ảnh thịt cần phân loại", 
                    help="Click vào nút camera để chụp ảnh",
                    key="camera_input"
                )
                
                if camera_photo is not None:
                    camera_image = Image.open(camera_photo)
                    st.image(camera_image, caption="Ảnh đã chụp", use_column_width=True)
                    
                    if st.button("Phân tích độ tươi", type="primary", key="camera_predict"):
                        analyze_image(model, camera_image, col2)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if not st.session_state.camera_enabled:
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Kết quả phân loại</div></div>', unsafe_allow_html=True)
                st.info("Vui lòng bật camera ở cột bên trái để bắt đầu")
                
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
                
                <div style="background: #fffbeb; padding: 1.25rem; border-radius: 6px; margin-top: 1rem; border: 1px solid #fef3c7;">
                    <strong style="color: #92400e;">Lưu ý:</strong> Camera chỉ bật khi cần để tiết kiệm tài nguyên hệ thống.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif camera_photo is None:
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Kết quả phân loại</div></div>', unsafe_allow_html=True)
                st.info("Vui lòng chụp ảnh ở cột bên trái để bắt đầu phân tích")
                
                st.markdown("""
                <div class="tips-section">
                    <div class="tips-title">Camera đã sẵn sàng!</div>
                    <ul class="tips-list">
                        <li>Click vào nút camera để chụp ảnh</li>
                        <li>Có thể "Tắt Camera" khi không dùng</li>
                    </ul>
                </div>
                
                <div style="background: #f0fdf4; padding: 1.25rem; border-radius: 6px; margin-top: 1rem; border: 1px solid #bbf7d0;">
                    <strong style="color: #166534;">Sẵn sàng phân tích!</strong><br>
                    Chất lượng ảnh tốt sẽ cho kết quả chính xác hơn.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer - Professional styling
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <div class="footer-brand">MeatFresh AI</div>
            <div class="footer-text">
                Hệ thống phân loại độ tươi thịt thông minh sử dụng Deep Learning<br>
                © 2024 - Phát triển bởi DW-SPPF Team
            </div>
            <div class="footer-disclaimer">
                Lưu ý: Kết quả phân tích mang tính chất tham khảo. Luôn kiểm tra thực tế trước khi sử dụng.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
