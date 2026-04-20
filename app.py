import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================================================
# CẤU HÌNH TRANG & CUSTOM CSS PROFESSIONAL
# ============================================================================

st.set_page_config(
    page_title="MeatFresh AI - Hệ thống phân loại độ tươi thịt thông minh",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện thương mại, hiện đại
CUSTOM_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

/* Main container styling */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 0;
    border-radius: 0 0 30px 30px;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    margin-bottom: 2rem;
}

.header-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    text-align: center;
}

.header-title {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.header-subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    font-weight: 300;
}

/* Card styling */
.card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1.5rem;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 48px rgba(0,0,0,0.15);
}

.card-title {
    color: #667eea;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Result box styling */
.result-box {
    background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    border: 2px solid #e0e7ff;
}

.result-box.success {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-color: #28a745;
}

.result-box.warning {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border-color: #ffc107;
}

.result-box.error {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-color: #dc3545;
}

.result-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.result-confidence {
    font-size: 1.2rem;
    color: #666;
    font-weight: 500;
}

/* Progress bar customization */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: white !important;
}

/* Info box styling */
.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid #2196f3;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.success-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-left: 4px solid #4caf50;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.warning-box {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-left: 4px solid #ff9800;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.error-box {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border-left: 4px solid #f44336;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    justify-content: center;
}

.stTabs [data-baseweb="tab"] {
    padding: 1rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    border: 2px dashed #667eea;
    border-radius: 15px;
    padding: 2rem;
    background: rgba(102, 126, 234, 0.05);
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #764ba2;
    background: rgba(102, 126, 234, 0.1);
}

/* Footer */
.footer {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
    color: white;
    padding: 2rem 0;
    margin-top: 3rem;
    border-radius: 30px 30px 0 0;
    text-align: center;
}

.footer-text {
    color: rgba(255,255,255,0.7);
    font-size: 0.9rem;
}

/* Animation for loading */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading-text {
    animation: pulse 1.5s ease-in-out infinite;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
}

.metric-label {
    color: #666;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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
                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600; color: #333;">{cn_vi} ({cn})</span>
                            <span style="color: #666;">{prob:.2%}</span>
                        </div>
                        <div style="background: #e0e7ff; border-radius: 10px; height: 12px; overflow: hidden;">
                            <div style="background: {bar_color}; width: {prob*100}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
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
    
    # Stats bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}</div>
            <div class="metric-label">Độ phân giải</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(CLASS_NAMES)}</div>
            <div class="metric-label">Lớp phân loại</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">DL</div>
            <div class="metric-label">Công nghệ AI</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">< 1s</div>
            <div class="metric-label">Thời gian xử lý</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Sidebar thông tin - không dùng emoji
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h3 style="color: white; margin-top: 1rem;">Thông tin hệ thống</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box" style="background: rgba(255,255,255,0.1); border-left: 4px solid white;">
            <strong style="color: white;">Kích thước đầu vào:</strong><br>
            <span style="color: rgba(255,255,255,0.9);">{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]} pixels</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 2rem; color: white;">
            <h4 style="border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 0.5rem;">Các lớp phân loại</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0; padding: 1rem; background: rgba(40, 167, 69, 0.2); border-radius: 10px;">
            <div style="width: 2rem; height: 2rem; border-radius: 50%; background: #28a745; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">✓</div>
            <div>
                <div style="font-weight: 600; color: white;">Tươi (Fresh)</div>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8);">Sản phẩm chất lượng tốt nhất</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0; padding: 1rem; background: rgba(255, 193, 7, 0.2); border-radius: 10px;">
            <div style="width: 2rem; height: 2rem; border-radius: 50%; background: #ffc107; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">!</div>
            <div>
                <div style="font-weight: 600; color: white;">Bán tươi (Half)</div>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8);">Cần sử dụng sớm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 1rem; margin: 1rem 0; padding: 1rem; background: rgba(220, 53, 69, 0.2); border-radius: 10px;">
            <div style="width: 2rem; height: 2rem; border-radius: 50%; background: #dc3545; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">✕</div>
            <div>
                <div style="font-weight: 600; color: white;">Hỏng (Spoiled)</div>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8);">Không nên sử dụng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 2rem; color: white;">
            <h4 style="border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 0.5rem;">Hướng dẫn</h4>
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
    
    # Main content - Tabs cho Upload và Camera - không dùng emoji
    tab1, tab2 = st.tabs(["Upload ảnh", "Chụp ảnh từ Camera"])
    
    # Tab 1: Upload ảnh
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card"><div class="card-title">Tải ảnh lên</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="card"><div class="card-title">Kết quả phân loại</div>', unsafe_allow_html=True)
                st.info("Vui lòng tải ảnh lên ở cột bên trái để bắt đầu phân tích")
                
                st.markdown("""
                ### Mẹo để có kết quả tốt nhất:
                - Sử dụng ảnh có độ phân giải cao
                - Đảm bảo ánh sáng đủ và đều
                - Thịt nên được chụp rõ nét, chiếm phần lớn khung hình
                - Tránh bóng đổ che khuất bề mặt thịt
                - Nên chụp từ góc nhìn trực diện
                
                <div style="background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%); padding: 1.5rem; border-radius: 15px; margin-top: 1rem; border: 2px dashed #667eea;">
                    <strong>Lưu ý:</strong> Kết quả phân tích mang tính chất tham khảo. 
                    Luôn kiểm tra thêm bằng các giác quan (mùi, màu sắc, kết cấu) trước khi sử dụng.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Chụp ảnh từ camera
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card"><div class="card-title">Chụp ảnh trực tiếp</div>', unsafe_allow_html=True)
            
            if 'camera_enabled' not in st.session_state:
                st.session_state.camera_enabled = False
            
            if not st.session_state.camera_enabled:
                if st.button("Bật Camera", type="primary", key="enable_camera"):
                    st.session_state.camera_enabled = True
                    st.rerun()
                
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: #f8f9ff; border-radius: 15px; border: 2px dashed #667eea;">
                    <div style="font-size: 3rem; margin-bottom: 1rem; color: #667eea;">📸</div>
                    <p style="color: #666;">Click nút "Bật Camera" để mở camera và chụp ảnh</p>
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
                st.markdown('<div class="card"><div class="card-title">Kết quả phân loại</div>', unsafe_allow_html=True)
                st.info("Vui lòng bật camera ở cột bên trái để bắt đầu")
                
                st.markdown("""
                ### Hướng dẫn chụp ảnh:
                1. **Đặt thịt** trên nền sáng, phẳng
                2. **Giữ camera ổn định** khi chụp
                3. **Đảm bảo ánh sáng** đủ sáng và đều
                4. **Chụp từ góc nhìn trực diện**
                5. **Tránh phản quang** và bóng đổ
                
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 1.5rem; border-radius: 15px; margin-top: 1rem; border-left: 4px solid #ffc107;">
                    <strong>Lưu ý:</strong> Camera chỉ bật khi cần để tiết kiệm tài nguyên hệ thống.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif camera_photo is None:
                st.markdown('<div class="card"><div class="card-title">Kết quả phân loại</div>', unsafe_allow_html=True)
                st.info("Vui lòng chụp ảnh ở cột bên trái để bắt đầu phân tích")
                
                st.markdown("""
                ### Camera đã sẵn sàng!
                - Click vào nút camera để chụp ảnh
                - Có thể "Tắt Camera" khi không dùng
                
                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1.5rem; border-radius: 15px; margin-top: 1rem; border-left: 4px solid #28a745;">
                    <strong>Sẵn sàng phân tích!</strong><br>
                    Chất lượng ảnh tốt sẽ cho kết quả chính xác hơn.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <div style="font-size: 1.5rem; margin-bottom: 1rem;">MeatFresh AI</div>
            <div class="footer-text">
                Hệ thống phân loại độ tươi thịt thông minh sử dụng Deep Learning<br>
                © 2024 - Phát triển bởi DW-SPPF Team
            </div>
            <div style="margin-top: 1rem; font-size: 0.85rem; color: rgba(255,255,255,0.5);">
                Lưu ý: Kết quả phân tích mang tính chất tham khảo. Luôn kiểm tra thực tế trước khi sử dụng.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
