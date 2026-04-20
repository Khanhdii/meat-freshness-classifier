import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================================================
# CẤU HÌNH TRANG & CUSTOM CSS - PROFESSIONAL DESIGN
# ============================================================================

st.set_page_config(
    page_title="MonFresh - AI Meat Freshness Analysis",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional color palette matching index.html
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header styling */
    .header-container {
        background: white;
        border-bottom: 1px solid #e2e8f0;
        padding: 1.25rem 2rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-content {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .header-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
    }
    
    .header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    
    .header-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Stats container */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -8px rgba(0, 0, 0, 0.08);
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Card styling */
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        margin-bottom: 1.25rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
    }
    
    /* Result boxes */
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid;
        text-align: center;
    }
    
    .result-box.success {
        background: #f0fdf4;
        border-color: #bbf7d0;
    }
    
    .result-box.warning {
        background: #fffbeb;
        border-color: #fef3c7;
    }
    
    .result-box.error {
        background: #fef2f2;
        border-color: #fecaca;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-confidence {
        font-size: 1rem;
        font-weight: 600;
        color: #475569;
    }
    
    /* Progress bars */
    .progress-container {
        background: #e2e8f0;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    
    /* Info/recommendation boxes */
    .success-box, .warning-box, .error-box {
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        line-height: 1.6;
    }
    
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        color: #166534;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: white;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }
    
    .info-box {
        background: rgba(255,255,255,0.1);
        border-left: 4px solid white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    
    .class-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .class-indicator {
        width: 24px;
        height: 24px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    
    .class-info-title {
        font-weight: 600;
        color: white;
        font-size: 0.875rem;
    }
    
    .class-info-desc {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
    }
    
    /* Tips section */
    .tips-section {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .tips-title {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
    }
    
    .tips-list {
        margin: 0;
        padding-left: 1.25rem;
        color: #475569;
        font-size: 0.875rem;
        line-height: 1.8;
    }
    
    .tips-list li {
        margin-bottom: 0.25rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2rem;
        color: #64748b;
    }
    
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .empty-state-title {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-desc {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* File uploader styling */
    .stFileUploader {
        margin-bottom: 1rem;
    }
    
    .stFileUploader > div {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #0ea5e9;
        background: rgba(14, 165, 233, 0.02);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button[type="primary"] {
        background: #0ea5e9;
        color: white;
        border: none;
    }
    
    .stButton > button[type="primary"]:hover {
        background: #0284c7;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0ea5e9;
        color: white;
    }
    
    /* Footer */
    .footer {
        background: white;
        border-top: 1px solid #e2e8f0;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .footer-text {
        font-size: 0.875rem;
        color: #64748b;
        line-height: 1.6;
    }
    
    .footer-disclaimer {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 1rem;
        font-style: italic;
    }
    
    /* Image container */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stImage img {
        border-radius: 12px;
    }
    
    /* Alert/Info messages */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Remove default Streamlit padding issues */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Column spacing */
    .element-container {
        margin-bottom: 1rem;
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
            <div class="header-logo">
                <div class="logo-icon">🥩</div>
                <div>
                    <div class="header-title">MonFresh</div>
                    <div class="header-subtitle">AI-Powered Meat Freshness Analysis</div>
                </div>
            </div>
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
            <div class="footer-brand">MonFresh</div>
            <div class="footer-text">
                AI-Powered Meat Freshness Analysis System<br>
                © 2024 - Powered by DW-SPPF Deep Learning Technology
            </div>
            <div class="footer-disclaimer">
                Note: Analysis results are for reference only. Always perform real-world inspection before use.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
