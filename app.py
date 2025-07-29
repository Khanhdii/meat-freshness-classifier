import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống phân loại độ tươi thịt",
    page_icon="🥩",
    layout="wide"
)

# Constants từ model training
INPUT_SHAPE = (224, 224, 3)
CLASS_NAMES = {0: 'FRESH', 1: 'HALF', 2: 'SPOILED'}
CLASS_NAMES_VI = {0: 'Tươi', 1: 'Bán tươi', 2: 'Hỏng'}

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

def analyze_image(model, image, col):
    """Phân tích ảnh và hiển thị kết quả"""
    with st.spinner("Đang phân tích..."):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with col:
                st.header("📊 Kết quả phân loại")
                
                # Kết quả chính
                class_name = CLASS_NAMES[predicted_class]
                class_name_vi = CLASS_NAMES_VI[predicted_class]
                
                # Chọn màu và icon theo kết quả
                if predicted_class == 0:  # Fresh
                    color = "🟢"
                    status_color = "success"
                elif predicted_class == 1:  # Half
                    color = "🟡" 
                    status_color = "warning"
                else:  # Spoiled
                    color = "🔴"
                    status_color = "error"
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: var(--secondary-background-color); text-align: center;">
                    <h2>{color} {class_name_vi}</h2>
                    <h3>({class_name})</h3>
                    <h4>Độ tin cậy: {confidence:.2%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Biểu đồ chi tiết các xác suất
                st.subheader("📈 Chi tiết xác suất")
                
                for i, (class_id, prob) in enumerate(zip(CLASS_NAMES.keys(), all_predictions)):
                    class_name = CLASS_NAMES[class_id]
                    class_name_vi = CLASS_NAMES_VI[class_id]
                    
                    # Progress bar với màu sắc
                    if i == 0:
                        st.success(f"🟢 {class_name_vi} ({class_name})")
                    elif i == 1:
                        st.warning(f"🟡 {class_name_vi} ({class_name})")
                    else:
                        st.error(f"🔴 {class_name_vi} ({class_name})")
                    
                    st.progress(float(prob))
                    st.write(f"Xác suất: {prob:.4f} ({prob:.2%})")
                    st.write("")
                
                # Khuyến nghị
                st.subheader("💡 Khuyến nghị")
                if predicted_class == 0:
                    st.success("✅ Thịt còn tươi, có thể sử dụng an toàn.")
                elif predicted_class == 1:
                    st.warning("⚠️ Thịt bán tươi, nên sử dụng sớm hoặc kiểm tra kỹ.")
                else:
                    st.error("❌ Thịt đã hỏng, không nên sử dụng.")
        
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

def main():
    st.title("🥩 Hệ thống phân loại độ tươi thịt")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Không thể load model. Vui lòng kiểm tra file model.")
        return
    
    st.success("✅ Model đã được load thành công!")
    
    # Sidebar thông tin
    with st.sidebar:
        st.header("📋 Thông tin model")
        st.info(f"""
        **Kích thước đầu vào:** {INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}
        **Số lớp phân loại:** {len(CLASS_NAMES)}
        **Các lớp:**
        - 🟢 Fresh (Tươi)
        - 🟡 Half (Bán tươi) 
        - 🔴 Spoiled (Hỏng)
        """)
        
        st.header("📖 Hướng dẫn sử dụng")
        st.markdown("""
        **📤 Tab Upload:**
        1. Chọn ảnh từ thiết bị của bạn
        2. Click "Phân tích độ tươi"
        3. Xem kết quả chi tiết
        
        **📷 Tab Camera:**
        1. Click "Bật Camera" để kích hoạt
        2. Chụp ảnh thịt cần phân loại
        3. Click "Phân tích độ tươi"
        4. Click "Tắt Camera" khi xong
        
        💡 **Tiết kiệm tài nguyên**: Camera chỉ bật khi cần
        """)
    
    # Main content - Thêm tabs cho Upload và Camera
    tab1, tab2 = st.tabs(["📤 Upload ảnh", "📷 Chụp ảnh"])
    
    # Tab 1: Upload ảnh
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📤 Upload ảnh từ thiết bị")
            uploaded_file = st.file_uploader(
                "Chọn ảnh thịt cần phân loại",
                type=['png', 'jpg', 'jpeg'],
                help="Hỗ trợ định dạng PNG, JPG, JPEG"
            )
        
            if uploaded_file is not None:
                # Hiển thị ảnh đã upload
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh đã upload", use_column_width=True)
                
                # Nút dự đoán
                if st.button("🔍 Phân tích độ tươi (Upload)", type="primary", key="upload_predict"):
                    analyze_image(model, image, col2)
        
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

if __name__ == "__main__":
    main() 