import streamlit as st
import numpy as np
from PIL import Image
import io
import random
import time
import tensorflow as tf

# Cấu hình trang
st.set_page_config(
    page_title="MONFRESH - Đánh giá độ tươi thịt bằng AI",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS tùy chỉnh cho MONFRESH
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #007bff, #28a745);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-section {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cta-button {
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        padding: 12px 30px;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,123,255,0.4);
    }
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .footer {
        background: #343a40;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants từ model training
INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES = {0: 'FRESH', 1: 'HALF', 2: 'SPOILED'}
CLASS_NAMES_VI = {0: 'Tươi', 1: 'Sắp hư', 2: 'Hỏng'}
CLASS_NAMES_LA = {0: 'ສົດ', 1: 'ໃກ້ເສຍ', 2: 'ເສຍ'}
CLASS_NAMES_KH = {0: 'ស្រស់', 1: 'ជិតខូច', 2: 'ខូច'}

@st.cache_resource
def load_model():
    """Load model đã được huấn luyện"""
    try:
        model = tf.keras.models.load_model('TinyYolo_model.keras')
        return model
    except:
        try:
            model = tf.keras.models.load_model('TinyYolo_model.h5')
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

def analyze_image_monfresh(model, image, col, class_names):
    """Phân tích ảnh và hiển thị kết quả với style MONFRESH"""
    with st.spinner("🤖 AI đang phân tích ảnh..."):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with col:
                st.markdown("### 📊 Kết quả phân loại")
                
                # Kết quả chính với style MONFRESH
                class_name = class_names[predicted_class]
                
                # Chọn màu và icon theo kết quả
                if predicted_class == 0:  # Fresh
                    color = "🟢"
                    emoji = "😊"
                    bg_color = "#d4edda"
                    text_color = "#155724"
                    recommendation = "✅ Thịt còn tươi, có thể sử dụng an toàn."
                elif predicted_class == 1:  # Half
                    color = "🟡" 
                    emoji = "😰"
                    bg_color = "#fff3cd"
                    text_color = "#856404"
                    recommendation = "⚠️ Thịt sắp hư, nên sử dụng sớm hoặc kiểm tra kỹ."
                else:  # Spoiled
                    color = "🔴"
                    emoji = "🤢"
                    bg_color = "#f8d7da"
                    text_color = "#721c24"
                    recommendation = "❌ Thịt đã hỏng, không nên sử dụng."
                
                # Kết quả chính
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 15px; background-color: {bg_color}; color: {text_color}; text-align: center; margin: 10px 0;">
                    <h2>{emoji} {class_name}</h2>
                    <h3>Độ tin cậy: {confidence:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Thanh màu theo cấp độ
                st.markdown("### 📈 Chi tiết xác suất")
                
                for i, (class_id, prob) in enumerate(zip(class_names.keys(), all_predictions)):
                    class_name = class_names[class_id]
                    
                    # Progress bar với màu sắc
                    if i == 0:
                        st.success(f"🟢 {class_name}")
                    elif i == 1:
                        st.warning(f"🟡 {class_name}")
                    else:
                        st.error(f"🔴 {class_name}")
                    
                    st.progress(float(prob))
                    st.write(f"**{prob:.1%}**")
                    st.write("")
                
                # Khuyến nghị
                st.markdown("### 💡 Khuyến nghị")
                st.info(recommendation)
                
                # QR Code và chia sẻ (placeholder)
                st.markdown("### 🔗 Chia sẻ kết quả")
                col_qr1, col_qr2 = st.columns(2)
                with col_qr1:
                    st.button("📱 Gửi qua Zalo", key="zalo_share")
                with col_qr2:
                    st.button("📧 Gửi qua Email", key="email_share")
        
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

def analyze_image(model, image, col):
    """Phân tích ảnh và hiển thị kết quả (legacy)"""
    analyze_image_monfresh(model, image, col, CLASS_NAMES_VI)

def main():
    # Header với logo MONFRESH
    st.markdown("""
    <div class="main-header">
        <h1>🥩 MONFRESH</h1>
        <p><strong>Chuẩn hóa độ tươi – Nâng tầm thực phẩm</strong></p>
        <p>Đánh giá độ tươi của thịt trong vài giây bằng AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h2>🔍 Công nghệ không chạm: an toàn – minh bạch – đơn giản</h2>
        <p>Chỉ cần một bức ảnh để biết thịt còn tươi hay không!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Không thể load model. Vui lòng kiểm tra file model.")
        return
    
    # Language selector
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        language = st.selectbox("🌐 Ngôn ngữ", ["🇻🇳 Tiếng Việt", "🇬🇧 English", "🇱🇦 ພາສາລາວ", "🇰🇭 ភាសាខ្មែរ"])
    
    # Map language to class names
    if "Tiếng Việt" in language:
        class_names = CLASS_NAMES_VI
    elif "English" in language:
        class_names = CLASS_NAMES
    elif "ພາສາລາວ" in language:
        class_names = CLASS_NAMES_LA
    else:
        class_names = CLASS_NAMES_KH
    
    # Main content - Thêm tabs cho Upload và Camera
    tab1, tab2 = st.tabs(["📤 Upload ảnh", "📷 Chụp ảnh"])
    
    # Tab 1: Upload ảnh
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload ảnh từ thiết bị")
            st.markdown("**Kéo & thả hoặc chọn ảnh thịt cần kiểm tra**")
            uploaded_file = st.file_uploader(
                "Chọn ảnh thịt cần phân loại",
                type=['png', 'jpg', 'jpeg'],
                help="Hỗ trợ định dạng PNG, JPG, JPEG"
            )
        
            if uploaded_file is not None:
                # Hiển thị ảnh đã upload
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh đã upload", use_column_width=True)
                
                # Nút dự đoán với style MONFRESH
                if st.button("🔍 Phân tích độ tươi", type="primary", key="upload_predict", use_container_width=True):
                    analyze_image_monfresh(model, image, col2, class_names)
        
        with col2:
            if uploaded_file is None:
                st.header("📊 Kết quả phân loại")
                st.info("👆 Vui lòng upload ảnh ở cột bên trái để bắt đầu phân tích")
                
                # Hiển thị ảnh mẫu hoặc hướng dẫn
                st.markdown("""
                ### 🎯 Mẹo để có kết quả tốt nhất:
                - Sử dụng ảnh có độ phân giải cao
                - Đảm bảo ánh sáng đủ và đều
                - Thịt nên được chụp rõ nét
                - Tránh bóng đổ che khuất
                """)
    
    # Tab 2: Chụp ảnh từ camera
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📷 Chụp ảnh từ camera")
            
            # Khởi tạo session state cho camera
            if 'camera_enabled' not in st.session_state:
                st.session_state.camera_enabled = False
            
            # Nút để bật/tắt camera
            if not st.session_state.camera_enabled:
                if st.button("📷 Bật Camera", type="primary", key="enable_camera"):
                    st.session_state.camera_enabled = True
                    st.rerun()
                st.info("👆 Click nút 'Bật Camera' để mở camera và chụp ảnh")
            else:
                # Hiển thị nút tắt camera
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if st.button("📷 Tắt Camera", key="disable_camera"):
                        st.session_state.camera_enabled = False
                        st.rerun()
                with col_b:
                    if st.button("🔄 Chụp ảnh mới", key="new_photo"):
                        # Reset camera để chụp ảnh mới
                        pass
                
                # Camera input chỉ hiển thị khi được bật
                camera_photo = st.camera_input(
                    "Chụp ảnh thịt cần phân loại", 
                    help="Click vào nút camera để chụp ảnh",
                    key="camera_input"
                )
                
                if camera_photo is not None:
                    # Hiển thị ảnh đã chụp
                    camera_image = Image.open(camera_photo)
                    st.image(camera_image, caption="Ảnh đã chụp", use_column_width=True)
                    
                    # Nút dự đoán
                    if st.button("🔍 Phân tích độ tươi (Camera)", type="primary", key="camera_predict"):
                        analyze_image(model, camera_image, col2)
        
        with col2:
            if not st.session_state.camera_enabled:
                st.header("📊 Kết quả phân loại")
                st.info("👆 Vui lòng bật camera ở cột bên trái để bắt đầu")
                
                # Hướng dẫn chụp ảnh
                st.markdown("""
                ### 📷 Hướng dẫn sử dụng camera:
                1. **Click "Bật Camera"** để kích hoạt camera
                2. **Đặt thịt** trên nền sáng, phẳng
                3. **Giữ camera ổn định** khi chụp
                4. **Đảm bảo ánh sáng** đủ sáng và đều
                5. **Chụp từ góc nhìn trực diện**
                6. **Tránh phản quang** và bóng đổ che khuất
                
                💡 **Lưu ý**: Camera chỉ bật khi cần để tiết kiệm tài nguyên
                """)
            elif camera_photo is None:
                st.header("📊 Kết quả phân loại")
                st.info("👆 Vui lòng chụp ảnh ở cột bên trái để bắt đầu phân tích")
                
                st.markdown("""
                ### ✅ Camera đã sẵn sàng!
                - Click vào nút camera để chụp ảnh
                - Có thể "Tắt Camera" khi không dùng để tiết kiệm tài nguyên
                """)
    
    # Footer MONFRESH
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>🥩 MONFRESH - Chuẩn hóa độ tươi – Nâng tầm thực phẩm</h3>
        <p><strong>MONFRESH</strong> là một nền tảng công nghệ ứng dụng trí tuệ nhân tạo (AI) giúp kiểm tra độ tươi của thịt một cách nhanh chóng, khách quan và dễ sử dụng – chỉ bằng một bức ảnh chụp từ điện thoại.</p>
        
        <h4>🔍 MONFRESH hoạt động như thế nào?</h4>
        <ul>
            <li>Người dùng chỉ cần truy cập web/app MONFRESH, chụp ảnh miếng thịt bằng camera điện thoại.</li>
            <li>Hệ thống AI sẽ phân tích ảnh và phân loại thịt thành 3 cấp độ: Tươi – Sắp hư – Hư hỏng.</li>
            <li>Mỗi lần kiểm tra được lưu kèm thời gian, vị trí, ảnh gốc và kết quả → tạo thành hồ sơ độ tươi có thể truy xuất.</li>
        </ul>
        
        <h4>🎯 Đối tượng sử dụng</h4>
        <ul>
            <li>Tiểu thương tại chợ truyền thống cần công cụ chứng minh chất lượng.</li>
            <li>Người tiêu dùng trẻ ưu tiên thực phẩm an toàn và có thể truy xuất.</li>
            <li>Cơ quan quản lý VSATTP cần giám sát hiệu quả tại cấp phường/xã.</li>
            <li>Chuỗi siêu thị, nhà máy chế biến muốn tích hợp công nghệ AI giám sát đầu vào.</li>
        </ul>
        
        <h4>⚙️ Điểm nổi bật của MONFRESH</h4>
        <ul>
            <li>Không phá mẫu – Không cần thiết bị chuyên dụng – Không yêu cầu kỹ thuật viên.</li>
            <li>Chạy trực tiếp trên điện thoại hoặc web, dễ sử dụng, tiết kiệm chi phí.</li>
            <li>Dễ tích hợp với hệ thống bán hàng, truy xuất, thương mại điện tử và quản lý nhà nước.</li>
        </ul>
        
        <h4>👥 Nhóm phát triển</h4>
        <p>Dự án được thực hiện bởi nhóm MONFRESH, bao gồm các sinh viên, kỹ sư và chuyên gia liên ngành: AI, công nghệ thực phẩm, kinh doanh và quản lý dữ liệu. Đại diện nhóm dự án: Đặng Hoàng Khang.</p>
        
        <h4>🔗 Liên hệ</h4>
        <p>Fanpage: <a href="https://www.facebook.com/profile.php?id=61577355852837" target="_blank">MONFRESH Facebook</a></p>
        <p>Đối tác công nghệ: Industrial University of Ho Chi Minh City & Ecotech - TechFest Vietnam</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 