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

# CSS tùy chỉnh cho MONFRESH - Modern Flat Design
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: #ffffff;
    }
    
    /* Main Header - Modern gradient without shadow */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 0;
        color: white;
        text-align: center;
        margin-bottom: 0;
        border: none;
    }
    
    /* Hero Section - Clean flat design */
    .hero-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem;
        border-radius: 0;
        text-align: center;
        margin-bottom: 2rem;
        border: none;
    }
    
    /* Feature Cards - Flat design with borders instead of shadows */
    .feature-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-3px);
    }
    
    /* Result Card - Modern clean design */
    .result-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    /* CTA Button - Modern flat button */
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 32px;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        opacity: 0.95;
    }
    
    /* Navigation - Clean modern style */
    .nav-menu {
        background: white;
        padding: 1rem 2rem;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    
    .nav-link {
        text-decoration: none;
        color: #333;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .nav-link:hover {
        background: #f5f5f5;
        color: #667eea;
    }
    
    /* Footer - Modern dark theme */
    .footer {
        background: #2d3748;
        color: white;
        padding: 3rem 2rem;
        border-radius: 0;
        margin-top: 3rem;
        border: none;
    }
    
    /* Status Badge - Flat design */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        border: none;
    }
    
    .status-fresh {
        background: #48bb78;
        color: white;
    }
    
    .status-half {
        background: #ecc94b;
        color: #333;
    }
    
    .status-spoiled {
        background: #f56565;
        color: white;
    }
    
    /* Progress Bar - Modern flat style */
    .progress-container {
        background: #edf2f7;
        border-radius: 8px;
        overflow: hidden;
        height: 12px;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    
    /* Info Box - Clean bordered design */
    .info-box {
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 1rem 0;
        background: #f8fafc;
    }
    
    .info-success {
        border-left-color: #48bb78;
        background: #f0fff4;
    }
    
    .info-warning {
        border-left-color: #ecc94b;
        background: #fffff0;
    }
    
    .info-error {
        border-left-color: #f56565;
        background: #fff5f5;
    }
    
    /* Remove Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a1a1a1;
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

def analyze_image_monfresh(model, image, col, class_names, ui_text):
    """Phân tích ảnh và hiển thị kết quả với style MONFRESH - Modern Flat Design"""
    with st.spinner("🤖 AI đang phân tích ảnh..."):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with col:
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="color: #333; margin-bottom: 1.5rem; text-align: center;">{ui_text['result_title']}</h3>
                """, unsafe_allow_html=True)
                
                # Kết quả chính với style MONFRESH
                class_name = class_names[predicted_class]
                
                # Chọn màu và icon theo kết quả
                if predicted_class == 0:  # Fresh
                    status_class = "status-fresh"
                    emoji = "😊"
                    border_color = "#48bb78"
                    bg_light = "#f0fff4"
                    text_dark = "#2f855a"
                    recommendation = "✅ Thịt còn tươi, có thể sử dụng an toàn."
                    info_class = "info-success"
                elif predicted_class == 1:  # Half
                    status_class = "status-half"
                    emoji = "😰"
                    border_color = "#ecc94b"
                    bg_light = "#fffff0"
                    text_dark = "#975a16"
                    recommendation = "⚠️ Thịt sắp hư, nên sử dụng sớm hoặc kiểm tra kỹ."
                    info_class = "info-warning"
                else:  # Spoiled
                    status_class = "status-spoiled"
                    emoji = "🤢"
                    border_color = "#f56565"
                    bg_light = "#fff5f5"
                    text_dark = "#c53030"
                    recommendation = "❌ Thịt đã hỏng, không nên sử dụng."
                    info_class = "info-error"
                
                # Kết quả chính với modern badge
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background: {bg_light}; border: 3px solid {border_color}; border-radius: 16px; margin: 1.5rem 0;">
                    <span class="status-badge {status_class}" style="font-size: 1.5rem; padding: 0.75rem 2rem; margin-bottom: 1rem;">
                        {emoji} {class_name}
                    </span>
                    <p style="color: {text_dark}; font-size: 1.3rem; font-weight: 600; margin: 1rem 0 0 0;">
                        {ui_text['confidence']}: <span style="color: {border_color};">{confidence:.1%}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Chi tiết xác suất với progress bars hiện đại
                st.markdown(f"""
                <div style="margin: 1.5rem 0;">
                    <h4 style="color: #333; margin-bottom: 1rem; font-size: 1.1rem;">{ui_text['details_title']}</h4>
                """, unsafe_allow_html=True)
                
                for i, (class_id, prob) in enumerate(zip(class_names.keys(), all_predictions)):
                    label_name = class_names[class_id]
                    
                    if i == 0:
                        bar_color = "#48bb78"
                        icon = "🟢"
                    elif i == 1:
                        bar_color = "#ecc94b"
                        icon = "🟡"
                    else:
                        bar_color = "#f56565"
                        icon = "🔴"
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 14px;">
                            <span style="font-weight: 500; color: #333;">{icon} {label_name}</span>
                            <span style="font-weight: 600; color: {bar_color};">{prob:.1%}</span>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {prob*100}%; background: {bar_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Khuyến nghị với info box hiện đại
                st.markdown(f"""
                <div style="margin: 1.5rem 0;">
                    <h4 style="color: #333; margin-bottom: 1rem; font-size: 1.1rem;">{ui_text['recommendation_title']}</h4>
                    <div class="info-box {info_class}">
                        <p style="margin: 0; color: {text_dark}; font-weight: 500; line-height: 1.6;">{recommendation}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Nút chia sẻ hiện đại
                st.markdown(f"""
                <div style="margin: 1.5rem 0;">
                    <h4 style="color: #333; margin-bottom: 1rem; font-size: 1.1rem;">{ui_text['share_title']}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col_qr1, col_qr2 = st.columns(2)
                with col_qr1:
                    st.button(ui_text['zalo_btn'], key="zalo_share", use_container_width=True)
                with col_qr2:
                    st.button(ui_text['email_btn'], key="email_share", use_container_width=True)
                
                # Tùy chọn nâng cao: premium với card design
                st.markdown("""
                <div style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #f6f8fb 0%, #ebf1f6 100%); border-radius: 12px; border: 2px solid #e0e0e0;">
                    <h4 style="color: #667eea; margin-bottom: 1rem; font-size: 1.1rem;">⭐ Tùy chọn nâng cao (Premium)</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col_prem1, col_prem2, col_prem3 = st.columns(3)
                with col_prem1:
                    st.text_input("📝 Ghi chú sản phẩm", placeholder="VD: Thịt heo sáng 7h", key="product_note")
                with col_prem2:
                    st.button("🏷️ Gắn nhãn QR", key="qr_label", use_container_width=True)
                with col_prem3:
                    st.button("📄 Tải PDF", key="download_pdf", use_container_width=True)
        
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

def analyze_image(model, image, col):
    """Phân tích ảnh và hiển thị kết quả (legacy)"""
    ui_text_vi = {
        "result_title": "📊 Kết quả phân loại",
        "confidence": "Độ tin cậy",
        "details_title": "📈 Chi tiết xác suất",
        "recommendation_title": "💡 Khuyến nghị",
        "share_title": "🔗 Chia sẻ kết quả",
        "zalo_btn": "📱 Gửi qua Zalo",
        "email_btn": "📧 Gửi qua Email"
    }
    analyze_image_monfresh(model, image, col, CLASS_NAMES_VI, ui_text_vi)

def main():
    # Header với logo MONFRESH - Modern Design
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between; max-width: 1200px; margin: 0 auto;">
            <div style="display: flex; align-items: center;">
                <img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=" 
                     style="width: 60px; height: 60px; margin-right: 15px; border-radius: 12px; border: 3px solid rgba(255,255,255,0.3);">
                <div>
                    <h1 style="margin: 0; color: white; font-size: 2rem; letter-spacing: 1px;">🥩 MONFRESH</h1>
                    <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">Chuẩn hóa độ tươi – Nâng tầm thực phẩm</p>
                </div>
            </div>
            <div style="text-align: right; color: white;">
                <button class="cta-button" style="background: rgba(255,255,255,0.2); border: 2px solid rgba(255,255,255,0.5);">
                    ✅ Kiểm tra độ tươi ngay
                </button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu - Modern flat design
    st.markdown("""
    <div class="nav-menu">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
            <div style="display: flex; gap: 10px;">
                <a href="#" class="nav-link">🏠 Trang chủ</a>
                <a href="#" class="nav-link">📖 Hướng dẫn</a>
                <a href="#" class="nav-link">📊 Lịch sử</a>
                <a href="#" class="nav-link">👤 Tài khoản</a>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="color: #666; font-size: 14px;">🌐 Ngôn ngữ:</span>
                <select id="language-select" style="padding: 8px 12px; border: 2px solid #e0e0e0; border-radius: 8px; background: white; font-size: 14px;">
                    <option value="vi">🇻🇳 Việt Nam</option>
                    <option value="en">🇬🇧 English</option>
                    <option value="la">🇱🇦 ພາສາລາວ</option>
                    <option value="kh">🇰🇭 ភាសាខ្មែរ</option>
                </select>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section với modern design
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="hero-section">
            <h2 style="color: white; font-size: 2.5rem; margin-bottom: 1rem; font-weight: 700;">🔍 {ui_text['hero_title']}</h2>
            <p style="color: rgba(255,255,255,0.95); font-size: 1.2rem; margin-bottom: 2rem; line-height: 1.6;">{ui_text['hero_subtitle']}</p>
            <button class="cta-button">
                🔎 Chụp ảnh / Upload ảnh ngay
            </button>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center; height: 100%;">
            <h4 style="color: #333; margin-bottom: 1.5rem; font-size: 1.1rem;">🎯 Kết quả AI phân tích realtime:</h4>
            <div style="display: flex; flex-direction: column; gap: 15px;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; padding: 15px; background: #f0fff4; border-radius: 12px; border: 2px solid #48bb78;">
                    <div style="width: 50px; height: 50px; background: #48bb78; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; font-weight: bold;">😊</div>
                    <div style="text-align: left;">
                        <p style="margin: 0; font-weight: 600; color: #2f855a; font-size: 16px;">Thịt Tươi</p>
                        <p style="margin: 3px 0 0 0; font-size: 12px; color: #666;">An toàn sử dụng</p>
                    </div>
                </div>
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; padding: 15px; background: #fffff0; border-radius: 12px; border: 2px solid #ecc94b;">
                    <div style="width: 50px; height: 50px; background: #ecc94b; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #333; font-size: 24px; font-weight: bold;">😰</div>
                    <div style="text-align: left;">
                        <p style="margin: 0; font-weight: 600; color: #975a16; font-size: 16px;">Sắp Hư</p>
                        <p style="margin: 3px 0 0 0; font-size: 12px; color: #666;">Nên sử dụng sớm</p>
                    </div>
                </div>
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; padding: 15px; background: #fff5f5; border-radius: 12px; border: 2px solid #f56565;">
                    <div style="width: 50px; height: 50px; background: #f56565; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; font-weight: bold;">🤢</div>
                    <div style="text-align: left;">
                        <p style="margin: 0; font-weight: 600; color: #c53030; font-size: 16px;">Thịt Hỏng</p>
                        <p style="margin: 3px 0 0 0; font-size: 12px; color: #666;">Không nên dùng</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Language selector (Streamlit native)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        language = st.selectbox("🌐 Ngôn ngữ", ["🇻🇳 Tiếng Việt", "🇬🇧 English", "🇱🇦 ພາສາລາວ", "🇰🇭 ភាសាខ្មែរ"])
    
    # Map language to class names and UI text
    if "Tiếng Việt" in language:
        class_names = CLASS_NAMES_VI
        ui_text = {
            "hero_title": "Đánh giá độ tươi của thịt trong vài giây bằng AI",
            "hero_subtitle": "Công nghệ không chạm: an toàn – minh bạch – đơn giản",
            "upload_title": "📤 Upload ảnh từ thiết bị",
            "upload_desc": "Kéo & thả hoặc chọn ảnh thịt cần kiểm tra",
            "camera_title": "📷 Chụp ảnh từ camera",
            "analyze_btn": "🔍 Phân tích độ tươi",
            "result_title": "📊 Kết quả phân loại",
            "confidence": "Độ tin cậy",
            "details_title": "📈 Chi tiết xác suất",
            "recommendation_title": "💡 Khuyến nghị",
            "share_title": "🔗 Chia sẻ kết quả",
            "zalo_btn": "📱 Gửi qua Zalo",
            "email_btn": "📧 Gửi qua Email"
        }
    elif "English" in language:
        class_names = CLASS_NAMES
        ui_text = {
            "hero_title": "Assess meat freshness in seconds with AI",
            "hero_subtitle": "Touchless technology: safe – transparent – simple",
            "upload_title": "📤 Upload image from device",
            "upload_desc": "Drag & drop or select meat image to check",
            "camera_title": "📷 Take photo with camera",
            "analyze_btn": "🔍 Analyze freshness",
            "result_title": "📊 Classification result",
            "confidence": "Confidence",
            "details_title": "📈 Probability details",
            "recommendation_title": "💡 Recommendation",
            "share_title": "🔗 Share result",
            "zalo_btn": "📱 Send via Zalo",
            "email_btn": "📧 Send via Email"
        }
    elif "ພາສາລາວ" in language:
        class_names = CLASS_NAMES_LA
        ui_text = {
            "hero_title": "ປະເມີນຄວາມສົດຂອງຊີ້ນໃນບັນທັດດ້ວຍ AI",
            "hero_subtitle": "ເທັກໂນໂລຊີບໍ່ສຳພັດ: ປອດໄພ – ໂປ່ງໃສ – ງ່າຍດາຍ",
            "upload_title": "📤 ອັບໂຫລດຮູບຈາກອຸປະກອນ",
            "upload_desc": "ລາກ & ວາງ ຫຼື ເລືອກຮູບຊີ້ນເພື່ອກວດສອບ",
            "camera_title": "📷 ຖ່າຍຮູບດ້ວຍກ້ອງ",
            "analyze_btn": "🔍 ວິເຄາະຄວາມສົດ",
            "result_title": "📊 ຜົນການຈັດປະເພດ",
            "confidence": "ຄວາມໝັ້ນໃຈ",
            "details_title": "📈 ລາຍລະອຽດຄວາມເປັນໄປໄດ້",
            "recommendation_title": "💡 ຄຳແນະນຳ",
            "share_title": "🔗 ແບ່ງປັນຜົນ",
            "zalo_btn": "📱 ສົ່ງຜ່ານ Zalo",
            "email_btn": "📧 ສົ່ງຜ່ານ Email"
        }
    else:  # Khmer
        class_names = CLASS_NAMES_KH
        ui_text = {
            "hero_title": "វាយតម្លៃភាពស្រស់របស់សាច់ក្នុងវិនាទីជាមួយ AI",
            "hero_subtitle": "បច្ចេកវិទ្យាមិនប៉ះ: សុវត្ថិភាព – ភាពច្បាស់លាស់ – ភាពងាយស្រួល",
            "upload_title": "📤 ផ្ទុករូបភាពឡើងពីឧបករណ៍",
            "upload_desc": "ទាញ & ដាក់ ឬជ្រើសរូបភាពសាច់ដើម្បីពិនិត្យ",
            "camera_title": "📷 ថតរូបជាមួយកាមេរ៉ា",
            "analyze_btn": "🔍 វិភាគភាពស្រស់",
            "result_title": "📊 លទ្ធផលចំណាត់ថ្នាក់",
            "confidence": "ភាពជឿជាក់",
            "details_title": "📈 ព័ត៌មានលម្អិតប្រូបាប៊ីលីធី",
            "recommendation_title": "💡 ការណែនាំ",
            "share_title": "🔗 ចែករំលែកលទ្ធផល",
            "zalo_btn": "📱 ផ្ញើតាមរយៈ Zalo",
            "email_btn": "📧 ផ្ញើតាមរយៈ Email"
        }
    
    # Main content - Thêm tabs cho Upload và Camera
    tab1, tab2 = st.tabs([ui_text['upload_title'], ui_text['camera_title']])
    
    # Tab 1: Upload ảnh
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"### {ui_text['upload_title']}")
            st.markdown(f"**{ui_text['upload_desc']}**")
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
                if st.button(ui_text['analyze_btn'], type="primary", key="upload_predict", use_container_width=True):
                    analyze_image_monfresh(model, image, col2, class_names, ui_text)
        
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
            st.markdown(f"### {ui_text['camera_title']}")
            
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
                if st.button(ui_text['analyze_btn'], type="primary", key="camera_predict"):
                    analyze_image_monfresh(model, camera_image, col2, class_names, ui_text)
        
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
    
    # Footer MONFRESH đầy đủ theo yêu cầu
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>🥩 MONFRESH - Chuẩn hóa độ tươi – Nâng tầm thực phẩm</h3>
        
        <p><strong>MONFRESH</strong> là một nền tảng công nghệ ứng dụng trí tuệ nhân tạo (AI) giúp kiểm tra độ tươi của thịt một cách nhanh chóng, khách quan và dễ sử dụng – chỉ bằng một bức ảnh chụp từ điện thoại.</p>
        
        <p>Dự án ra đời với mục tiêu giải quyết các vấn đề tồn đọng trong chuỗi cung ứng thực phẩm tươi sống như: đánh giá cảm quan thiếu chính xác, thiếu minh bạch trong truy xuất chất lượng, và sự hạn chế về công cụ kiểm định tại các điểm bán nhỏ lẻ, chợ truyền thống.</p>
        
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
        
        <h4>📈 Tác động xã hội & kinh tế</h4>
        <ul>
            <li>Giảm lãng phí thực phẩm do phát hiện sớm thịt hỏng.</li>
            <li>Tăng uy tín người bán nhờ minh bạch hóa chất lượng.</li>
            <li>Hỗ trợ số hóa chợ truyền thống và xây dựng hệ sinh thái thực phẩm an toàn – minh bạch – bền vững.</li>
        </ul>
        
        <h4>👥 Nhóm phát triển</h4>
        <p>Dự án được thực hiện bởi nhóm MONFRESH, bao gồm các sinh viên, kỹ sư và chuyên gia liên ngành: AI, công nghệ thực phẩm, kinh doanh và quản lý dữ liệu. Đại diện nhóm dự án: <strong>Đặng Hoàng Khang</strong>.</p>
        
        <h4>🔗 MONFRESH hướng đến trở thành một nền tảng kiểm định thực phẩm bằng AI phổ biến tại Việt Nam và mở rộng ra khu vực ASEAN trong tương lai gần.</h4>
        
        <h4>🧠 Project Introduction – MONFRESH</h4>
        <p><strong>Standardizing Freshness – Elevating Food Quality</strong></p>
        <p>MONFRESH is a technology platform that leverages artificial intelligence (AI) to assess the freshness of meat instantly and objectively—all through a single photo taken with a smartphone.</p>
        <p>The project was developed to address long-standing issues in the fresh food supply chain, such as unreliable sensory-based evaluations, lack of transparency in quality control, and the absence of effective inspection tools for small vendors and traditional markets.</p>
        
        <h4>🔍 How Does MONFRESH Work?</h4>
        <ul>
            <li>Users simply access the MONFRESH web or mobile app and take a photo of the meat using their phone camera.</li>
            <li>The AI system analyzes the image and classifies the meat into three levels: Fresh – Near Spoilage – Spoiled.</li>
            <li>Each inspection is logged with a timestamp, location, original photo, and result—creating a traceable freshness profile for every batch.</li>
        </ul>
        
        <h4>🎯 Target Users</h4>
        <ul>
            <li>Small-scale vendors in traditional markets needing a tool to verify product quality.</li>
            <li>Young consumers who prioritize safe and traceable food.</li>
            <li>Food safety authorities requiring efficient oversight tools at the local level.</li>
            <li>Supermarkets and processing plants looking to integrate AI for quality control at the input stage.</li>
        </ul>
        
        <h4>⚙️ Key Highlights of MONFRESH</h4>
        <ul>
            <li>No need for sample destruction – No specialized equipment – No technical expertise required.</li>
            <li>Runs directly on smartphones or web browsers, making it cost-effective and easy to use.</li>
            <li>Seamless integration with POS systems, traceability platforms, e-commerce, and public administration tools.</li>
        </ul>
        
        <h4>📈 Social & Economic Impact</h4>
        <ul>
            <li>Reduces food waste by detecting spoilage early.</li>
            <li>Enhances vendor credibility through quality transparency.</li>
            <li>Supports digital transformation in traditional markets and builds a sustainable, safe, and transparent food ecosystem.</li>
        </ul>
        
        <h4>👥 The Development Team</h4>
        <p>The project is led by MONFRESH, a multidisciplinary team of students, engineers, and experts in AI, food technology, business, and data management. Team representative: <strong>Đặng Hoàng Khang</strong>.</p>
        
        <h4>🔗 MONFRESH aspires to become the most widely adopted AI-based food inspection platform in Vietnam and expand across the ASEAN region in the near future.</h4>
        
        <h4>🔗 Liên hệ & Đối tác</h4>
        <p><strong>Đối tác công nghệ / truyền thông:</strong></p>
        <ul>
            <li>Industrial University of Ho Chi Minh City</li>
            <li>Ecotech - TechFest Vietnam</li>
        </ul>
        <p><strong>Liên hệ / mạng xã hội:</strong></p>
        <ul>
            <li>Fanpage: <a href="https://www.facebook.com/profile.php?id=61577355852837" target="_blank">MONFRESH Facebook</a></li>
            <li>Website: Bổ sung sau</li>
            <li>Tiktok: Bổ sung sau</li>
        </ul>
        <p><em>Chính sách bảo mật / điều khoản sử dụng: sau</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 