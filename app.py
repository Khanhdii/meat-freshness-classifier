import streamlit as st
import numpy as np
from PIL import Image
import io
import random
import time
import tensorflow as tf

# Cấu hình trang
st.set_page_config(
    page_title="MonFresh - AI Meat Freshness Detector",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS tùy chỉnh cho MonFresh - Premium Modern Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Styles - Premium White Theme */
    .stApp {
        background: linear-gradient(180deg, #FAFAFA 0%, #FFFFFF 100%);
    }
    
    /* Main Header - Elegant Gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    
    /* Hero Section - Modern Clean */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem auto;
        max-width: 1400px;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -5%;
        width: 200px;
        height: 200px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 50%;
    }
    
    /* Feature Cards - Glassmorphism Style */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        border-color: #667eea;
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    /* Result Card - Premium Design */
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    
    /* Primary Button - Modern Gradient */
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 40px;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.3px;
        position: relative;
        overflow: hidden;
    }
    
    .primary-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .primary-btn:hover::before {
        left: 100%;
    }
    
    .primary-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Secondary Button - Outline Style */
    .secondary-btn {
        background: transparent;
        color: #667eea;
        padding: 14px 32px;
        border-radius: 12px;
        border: 2px solid #667eea;
        font-weight: 600;
        font-size: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .secondary-btn:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Navigation - Minimalist */
    .nav-menu {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem 0;
        border-bottom: 1px solid #f3f4f6;
        margin-bottom: 2rem;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .nav-link {
        text-decoration: none;
        color: #4b5563;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        display: inline-block;
        margin: 0 0.25rem;
    }
    
    .nav-link:hover {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        color: #667eea;
        transform: translateY(-2px);
    }
    
    .nav-link.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer - Dark Premium */
    .footer {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 4rem 2rem 2rem;
        border-radius: 24px 24px 0 0;
        margin-top: 4rem;
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }
    
    /* Status Badge - Modern Pill */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .status-fresh {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .status-half {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .status-spoiled {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    /* Progress Bar - Animated Gradient */
    .progress-container {
        background: linear-gradient(90deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 12px;
        overflow: hidden;
        height: 14px;
        margin: 0.75rem 0;
        position: relative;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Info Box - Modern Alert */
    .info-box {
        padding: 1.75rem;
        border-radius: 16px;
        border-left: 5px solid;
        margin: 1.5rem 0;
        background: white;
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateX(5px);
    }
    
    .info-success {
        border-left-color: #10b981;
        background: linear-gradient(90deg, #f0fdf4 0%, #ffffff 100%);
    }
    
    .info-warning {
        border-left-color: #f59e0b;
        background: linear-gradient(90deg, #fffbeb 0%, #ffffff 100%);
    }
    
    .info-error {
        border-left-color: #ef4444;
        background: linear-gradient(90deg, #fef2f2 0%, #ffffff 100%);
    }
    
    /* Upload Zone - Dashed Border */
    .upload-zone {
        border: 3px dashed #d1d5db;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f9fafb 0%, #ffffff 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
        transform: scale(1.02);
    }
    
    /* Stat Card - Minimalist */
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #f3f4f6;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: #667eea;
        transform: translateY(-4px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Remove Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Smooth Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f9fafb;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
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
    """Phân tích ảnh và hiển thị kết quả với style MonFresh - Modern Flat Design"""
    with st.spinner("🤖 AI đang phân tích ảnh..."):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with col:
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="color: #333; margin-bottom: 1.5rem; text-align: center;">{ui_text['result_title']}</h3>
                """, unsafe_allow_html=True)
                
                # Kết quả chính với style MonFresh
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
    # Header với logo MonFresh - Premium Design
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between; max-width: 1200px; margin: 0 auto; padding: 0 2rem; position: relative; z-index: 1;">
            <div style="display: flex; align-items: center;">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #10b981 0%, #34d399 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 28px;">🥩</div>
                <div>
                    <h1 style="margin: 0; color: white; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.5px;">MonFresh</h1>
                    <p style="margin: 3px 0 0 0; color: rgba(255,255,255,0.7); font-size: 13px; font-weight: 400;">AI-Powered Meat Freshness Detector</p>
                </div>
            </div>
            <div style="display: flex; gap: 12px;">
                <button class="secondary-btn" style="background: transparent; color: white; border-color: rgba(255,255,255,0.3);">
                    📞 Liên hệ
                </button>
                <button class="primary-btn">
                    ✨ Dùng thử ngay
                </button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu - Minimalist Modern
    st.markdown("""
    <div class="nav-menu">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <div style="display: flex; gap: 8px;">
                <a href="#" class="nav-link active">🏠 Trang chủ</a>
                <a href="#" class="nav-link">📖 Hướng dẫn</a>
                <a href="#" class="nav-link">📊 Lịch sử</a>
                <a href="#" class="nav-link">👤 Tài khoản</a>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <select id="language-select" style="padding: 10px 16px; border: 1px solid #e5e7eb; border-radius: 10px; background: white; font-size: 14px; font-weight: 500; cursor: pointer; outline: none;">
                    <option value="vi">🇻🇳 Tiếng Việt</option>
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
            <h2 style="color: white; font-size: 2.5rem; margin-bottom: 1rem; font-weight: 700; position: relative; z-index: 1;">🔍 {ui_text['hero_title']}</h2>
            <p style="color: rgba(255,255,255,0.95); font-size: 1.2rem; margin-bottom: 2rem; line-height: 1.6; position: relative; z-index: 1;">{ui_text['hero_subtitle']}</p>
            <button class="primary-btn" style="position: relative; z-index: 1;">
                📸 Chụp ảnh / Upload ảnh ngay
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
                
                # Nút dự đoán với style MonFresh
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
    
    # Footer MonFresh đầy đủ theo yêu cầu
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>🥩 MonFresh - Chuẩn hóa độ tươi – Nâng tầm thực phẩm</h3>
        
        <p><strong>MonFresh</strong> là một nền tảng công nghệ ứng dụng trí tuệ nhân tạo (AI) giúp kiểm tra độ tươi của thịt một cách nhanh chóng, khách quan và dễ sử dụng – chỉ bằng một bức ảnh chụp từ điện thoại.</p>
        
        <p>Dự án ra đời với mục tiêu giải quyết các vấn đề tồn đọng trong chuỗi cung ứng thực phẩm tươi sống như: đánh giá cảm quan thiếu chính xác, thiếu minh bạch trong truy xuất chất lượng, và sự hạn chế về công cụ kiểm định tại các điểm bán nhỏ lẻ, chợ truyền thống.</p>
        
        <h4>🔍 MonFresh hoạt động như thế nào?</h4>
        <ul>
            <li>Người dùng chỉ cần truy cập web/app MonFresh, chụp ảnh miếng thịt bằng camera điện thoại.</li>
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
        
        <h4>⚙️ Điểm nổi bật của MonFresh</h4>
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
        <p>Dự án được thực hiện bởi nhóm MonFresh, bao gồm các sinh viên, kỹ sư và chuyên gia liên ngành: AI, công nghệ thực phẩm, kinh doanh và quản lý dữ liệu. Đại diện nhóm dự án: <strong>Đặng Hoàng Khang</strong>.</p>
        
        <h4>🔗 MonFresh hướng đến trở thành một nền tảng kiểm định thực phẩm bằng AI phổ biến tại Việt Nam và mở rộng ra khu vực ASEAN trong tương lai gần.</h4>
        
        <h4>🧠 Project Introduction – MonFresh</h4>
        <p><strong>Standardizing Freshness – Elevating Food Quality</strong></p>
        <p>MonFresh is a technology platform that leverages artificial intelligence (AI) to assess the freshness of meat instantly and objectively—all through a single photo taken with a smartphone.</p>
        <p>The project was developed to address long-standing issues in the fresh food supply chain, such as unreliable sensory-based evaluations, lack of transparency in quality control, and the absence of effective inspection tools for small vendors and traditional markets.</p>
        
        <h4>🔍 How Does MonFresh Work?</h4>
        <ul>
            <li>Users simply access the MonFresh web or mobile app and take a photo of the meat using their phone camera.</li>
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
        
        <h4>⚙️ Key Highlights of MonFresh</h4>
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
        <p>The project is led by MonFresh, a multidisciplinary team of students, engineers, and experts in AI, food technology, business, and data management. Team representative: <strong>Đặng Hoàng Khang</strong>.</p>
        
        <h4>🔗 MonFresh aspires to become the most widely adopted AI-based food inspection platform in Vietnam and expand across the ASEAN region in the near future.</h4>
        
        <h4>🔗 Liên hệ & Đối tác</h4>
        <p><strong>Đối tác công nghệ / truyền thông:</strong></p>
        <ul>
            <li>Industrial University of Ho Chi Minh City</li>
            <li>Ecotech - TechFest Vietnam</li>
        </ul>
        <p><strong>Liên hệ / mạng xã hội:</strong></p>
        <ul>
            <li>Fanpage: <a href="https://www.facebook.com/profile.php?id=61577355852837" target="_blank">MonFresh Facebook</a></li>
            <li>Website: Bổ sung sau</li>
            <li>Tiktok: Bổ sung sau</li>
        </ul>
        <p><em>Chính sách bảo mật / điều khoản sử dụng: sau</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 