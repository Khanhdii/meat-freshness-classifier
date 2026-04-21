import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================================================
# CẤU HÌNH TRANG & CUSTOM CSS - MODERN NATURE THEME
# ============================================================================

st.set_page_config(
    page_title="MonFresh - Phân Tích Độ Tươi Thịt Bằng AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern Nature Theme with Professional Layout
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Theme Color Variables - Modern Nature Palette */
    :root {
        --primary-50: #F0FDF4;
        --primary-100: #DCFCE7;
        --primary-200: #BBF7D0;
        --primary-300: #86EFAC;
        --primary-400: #4ADE80;
        --primary-500: #22C55E;
        --primary-600: #16A34A;
        --primary-700: #15803D;
        --primary-800: #166534;
        --primary-900: #14532D;
        
        --neutral-50: #FAFAFA;
        --neutral-100: #F5F5F5;
        --neutral-200: #E5E5E5;
        --neutral-300: #D4D4D4;
        --neutral-400: #A3A3A3;
        --neutral-500: #737373;
        --neutral-600: #525252;
        --neutral-700: #404040;
        --neutral-800: #262626;
        --neutral-900: #171717;
        
        --success: #16A34A;
        --warning: #CA8A04;
        --danger: #DC2626;
        
        --bg-main: #FAFAF9;
        --bg-card: #FFFFFF;
        --bg-sidebar: #F0FDF4;
    }
    
    /* Reset & Base */
    * {
        box-sizing: border-box;
    }
    
    .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, var(--bg-main) 0%, var(--primary-50) 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Main container adjustments */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px !important;
    }
    
    /* Header - Modern gradient design */
    .app-header {
        background: linear-gradient(135deg, var(--primary-700) 0%, var(--primary-600) 100%);
        padding: 2rem;
        margin: -1.5rem -1.5rem 2rem -1.5rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 4px 20px rgba(22, 163, 74, 0.15);
    }
    
    .app-header-content {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: 1rem;
        color: var(--primary-100);
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Stats Row */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--primary-100);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(22, 163, 74, 0.1);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-600);
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--neutral-500);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Content Cards */
    .content-card {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid var(--neutral-200);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    
    .card-heading {
        font-size: 1.125rem;
        font-weight: 700;
        color: var(--neutral-800);
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--primary-100);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .card-heading-dot {
        width: 8px;
        height: 8px;
        background: var(--primary-500);
        border-radius: 50%;
    }
    
    /* Result Display - Modern */
    .result-card {
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        border: 2px solid;
        transition: all 0.3s ease;
    }
    
    .result-card.fresh {
        background: linear-gradient(135deg, var(--primary-50) 0%, var(--primary-100) 100%);
        border-color: var(--primary-400);
    }
    
    .result-card.half {
        background: linear-gradient(135deg, #FEFCE8 0%, #FEF9C3 100%);
        border-color: #FDE047;
    }
    
    .result-card.spoiled {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-color: #FCA5A5;
    }
    
    .result-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }
    
    .result-badge.fresh {
        background: var(--primary-500);
        color: white;
    }
    
    .result-badge.half {
        background: var(--warning);
        color: white;
    }
    
    .result-badge.spoiled {
        background: var(--danger);
        color: white;
    }
    
    .result-main-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .result-main-text.fresh { color: var(--primary-700); }
    .result-main-text.half { color: #A16207; }
    .result-main-text.spoiled { color: #B91C1C; }
    
    .result-subtext {
        font-size: 0.875rem;
        color: var(--neutral-500);
        margin-bottom: 1.25rem;
    }
    
    .result-confidence {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--bg-card);
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--neutral-700);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Progress Bars */
    .progress-section {
        margin-top: 1.5rem;
    }
    
    .progress-item {
        margin-bottom: 1.25rem;
    }
    
    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .progress-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--neutral-700);
    }
    
    .progress-value {
        font-size: 0.875rem;
        font-weight: 700;
        color: var(--neutral-600);
    }
    
    .progress-track {
        height: 10px;
        background: var(--neutral-200);
        border-radius: 9999px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .progress-fill.fresh { background: linear-gradient(90deg, var(--primary-400), var(--primary-600)); }
    .progress-fill.half { background: linear-gradient(90deg, #FDE047, #F59E0B); }
    .progress-fill.spoiled { background: linear-gradient(90deg, #FCA5A5, #DC2626); }
    
    /* Recommendations */
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        font-size: 0.9375rem;
        line-height: 1.7;
        border-left: 4px solid;
    }
    
    .recommendation-card.fresh {
        background: var(--primary-50);
        border-left-color: var(--primary-500);
        color: var(--primary-800);
    }
    
    .recommendation-card.half {
        background: #FEFCE8;
        border-left-color: var(--warning);
        color: #854D0E;
    }
    
    .recommendation-card.spoiled {
        background: #FEF2F2;
        border-left-color: var(--danger);
        color: #991B1B;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--primary-200);
    }
    
    .sidebar-heading {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--primary-700);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--primary-200);
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.875rem;
        padding: 1rem;
        background: var(--bg-card);
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border: 1px solid var(--primary-100);
        transition: transform 0.2s ease;
    }
    
    .legend-item:hover {
        transform: translateX(4px);
    }
    
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    .legend-dot.fresh { background: var(--primary-500); }
    .legend-dot.half { background: var(--warning); }
    .legend-dot.spoiled { background: var(--danger); }
    
    .legend-content {
        flex: 1;
    }
    
    .legend-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--neutral-800);
        margin-bottom: 0.125rem;
    }
    
    .legend-desc {
        font-size: 0.75rem;
        color: var(--neutral-500);
    }
    
    /* Empty State - Beautiful Design */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, var(--primary-50) 0%, var(--bg-card) 100%);
        border-radius: 16px;
        border: 2px dashed var(--primary-200);
    }
    
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }
    
    .empty-state-title {
        font-size: 1.125rem;
        font-weight: 700;
        color: var(--neutral-700);
        margin-bottom: 0.5rem;
    }
    
    .empty-state-desc {
        font-size: 0.875rem;
        color: var(--neutral-500);
        line-height: 1.6;
    }
    
    /* Tips Section */
    .tips-card {
        background: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        margin-top: 1.25rem;
        border: 1px solid var(--neutral-200);
    }
    
    .tips-heading {
        font-size: 0.875rem;
        font-weight: 700;
        color: var(--neutral-700);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .tips-list {
        margin: 0;
        padding-left: 1.25rem;
        color: var(--neutral-600);
        font-size: 0.8125rem;
        line-height: 1.8;
    }
    
    .tips-list li {
        margin-bottom: 0.375rem;
    }
    
    /* Info Boxes */
    .info-box {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 0.8125rem;
        line-height: 1.6;
    }
    
    .info-box.blue {
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        color: #1E40AF;
    }
    
    .info-box.amber {
        background: #FFFBEB;
        border: 1px solid #FEF3C7;
        color: #92400E;
    }
    
    .info-box.green {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border: 2px dashed var(--neutral-300);
        border-radius: 12px;
        padding: 1.5rem;
        background: var(--bg-card);
        transition: border-color 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-400);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9375rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        border: none;
        font-family: inherit;
    }
    
    .stButton > button[type="primary"] {
        background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-700) 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.2);
    }
    
    .stButton > button[type="primary"]:hover {
        background: linear-gradient(135deg, var(--primary-700) 0%, var(--primary-800) 100%);
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:not([type="primary"]) {
        background: var(--bg-card);
        color: var(--neutral-700);
        border: 1px solid var(--neutral-300);
    }
    
    .stButton > button:not([type="primary"]):hover {
        background: var(--neutral-50);
        border-color: var(--neutral-400);
    }
    
    /* Tabs - Clean modern design without red slider */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        padding: 0;
        border-bottom: 2px solid var(--neutral-200);
        border-radius: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: var(--neutral-500);
        font-size: 0.9375rem;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary-600);
        background: var(--primary-50);
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: var(--primary-600);
        border-bottom-color: var(--primary-600);
        font-weight: 700;
    }
    
    /* Image Wrapper */
    .image-frame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--neutral-200);
        background: var(--bg-card);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Footer */
    .app-footer {
        background: var(--bg-card);
        border-top: 1px solid var(--neutral-200);
        padding: 1.5rem;
        margin-top: 3rem;
        text-align: center;
        color: var(--neutral-500);
        font-size: 0.8125rem;
        border-radius: 16px 16px 0 0;
    }
    
    /* Camera Input Styling */
    .stCameraInput > div {
        border: 2px dashed var(--neutral-300);
        border-radius: 12px;
        padding: 1.5rem;
        background: var(--bg-card);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .stats-row {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .app-title {
            font-size: 1.5rem;
        }
        
        .result-main-text {
            font-size: 1.75rem;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--neutral-100);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--neutral-300);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--neutral-400);
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
    """Phân tích ảnh và hiển thị kết quả theo phong cách hiện đại"""
    with st.spinner(""):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with result_col:
                class_name = CLASS_NAMES[predicted_class]
                class_name_vi = CLASS_NAMES_VI[predicted_class]
                result_type = ['fresh', 'half', 'spoiled'][predicted_class]
                
                # Result card chính
                st.markdown(f"""
                <div class="result-card {result_type}">
                    <div class="result-badge {result_type}">{class_name}</div>
                    <div class="result-main-text {result_type}">{class_name_vi}</div>
                    <div class="result-subtext">Kết quả phân tích từ AI</div>
                    <div class="result-confidence">
                        <span>✓</span> Độ tin cậy: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bars section
                st.markdown('<div class="content-card progress-section"><div class="card-heading"><div class="card-heading-dot"></div>Xác suất phân loại</div>', unsafe_allow_html=True)
                
                for class_id, prob in zip(CLASS_NAMES.keys(), all_predictions):
                    cn = CLASS_NAMES[class_id]
                    cn_vi = CLASS_NAMES_VI[class_id]
                    fill_type = ['fresh', 'half', 'spoiled'][class_id]
                    
                    st.markdown(f"""
                    <div class="progress-item">
                        <div class="progress-header">
                            <span class="progress-label">{cn_vi} ({cn})</span>
                            <span class="progress-value">{prob:.1%}</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill {fill_type}" style="width: {prob*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Khuyến nghị
                st.markdown('<div class="content-card"><div class="card-heading"><div class="card-heading-dot"></div>Khuyến nghị</div>', unsafe_allow_html=True)
                
                if predicted_class == 0:
                    st.markdown("""
                    <div class="recommendation-card fresh">
                        <strong>🟢 Thịt còn tươi</strong><br><br>
                        Sản phẩm ở trạng thái tốt nhất, có thể sử dụng an toàn ngay lập tức.
                        Nên bảo quản ở nhiệt độ 0-4°C để duy trì độ tươi lâu hơn.
                    </div>
                    """, unsafe_allow_html=True)
                elif predicted_class == 1:
                    st.markdown("""
                    <div class="recommendation-card half">
                        <strong>🟡 Thịt bán tươi</strong><br><br>
                        Sản phẩm vẫn có thể sử dụng nhưng nên chế biến sớm trong vòng 24 giờ.
                        Kiểm tra kỹ mùi và kết cấu trước khi sử dụng. Nấu chín kỹ trước khi ăn.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="recommendation-card spoiled">
                        <strong>🔴 Thịt đã hỏng</strong><br><br>
                        <strong>Không nên sử dụng sản phẩm này.</strong><br>
                        Có nguy cơ cao gây ngộ độc thực phẩm. Nên bỏ ngay để đảm bảo an toàn sức khỏe.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")

def main():
    # Modern gradient header
    st.markdown("""
    <div class="app-header">
        <div class="app-header-content">
            <div class="app-title">MonFresh</div>
            <div class="app-subtitle">Phân tích độ tươi của thịt bằng trí tuệ nhân tạo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Không thể load model. Vui lòng kiểm tra file model.")
        return
    
    # Stats row
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-number">{INPUT_SHAPE[0]}×{INPUT_SHAPE[1]}</div>
            <div class="stat-label">Độ phân giải</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(CLASS_NAMES)}</div>
            <div class="stat-label">Lớp phân loại</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">Deep Learning</div>
            <div class="stat-label">Công nghệ</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">&lt; 1s</div>
            <div class="stat-label">Thời gian xử lý</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar với legend hiện đại
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-heading">Phân loại kết quả</div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="legend-item">
            <div class="legend-dot fresh"></div>
            <div class="legend-content">
                <div class="legend-title">Tươi (FRESH)</div>
                <div class="legend-desc">Sản phẩm chất lượng tốt nhất</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="legend-item">
            <div class="legend-dot half"></div>
            <div class="legend-content">
                <div class="legend-title">Bán tươi (HALF)</div>
                <div class="legend-desc">Nên sử dụng sớm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="legend-item">
            <div class="legend-dot spoiled"></div>
            <div class="legend-content">
                <div class="legend-title">Hỏng (SPOILED)</div>
                <div class="legend-desc">Không nên sử dụng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-heading" style="margin-top: 2rem;">Hướng dẫn nhanh</div>
        <div style="font-size: 0.8125rem; line-height: 1.8; color: var(--neutral-600);">
            <div style="margin-bottom: 0.5rem;"><strong>1.</strong> Tải ảnh hoặc chụp từ camera</div>
            <div style="margin-bottom: 0.5rem;"><strong>2.</strong> Click nút "Phân tích độ tươi"</div>
            <div><strong>3.</strong> Xem kết quả ở cột bên phải</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content - Tabs cho Upload và Camera
    tab1, tab2 = st.tabs(["📤 Upload ảnh", "📷 Chụp từ Camera"])
    
    # Tab 1: Upload ảnh
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="content-card"><div class="card-heading"><div class="card-heading-dot"></div>Tải ảnh lên để phân tích</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Chọn file ảnh",
                type=['png', 'jpg', 'jpeg'],
                help="Hỗ trợ định dạng PNG, JPG, JPEG",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.markdown('<div style="margin-top: 1.5rem;" class="image-frame">', unsafe_allow_html=True)
                st.image(image, caption="", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                if st.button("🔍 Phân tích độ tươi", type="primary", key="upload_predict", use_container_width=True):
                    analyze_image(model, image, col2)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state" style="margin-top: 1.5rem;">
                    <div class="empty-state-title">Chưa chọn ảnh</div>
                    <div class="empty-state-desc">Vui lòng tải ảnh lên để bắt đầu phân tích độ tươi của thịt</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if uploaded_file is None:
                st.markdown('<div class="content-card"><div class="card-heading"><div class="card-heading-dot"></div>Kết quả phân tích</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-title">Chờ ảnh để phân tích</div>
                    <div class="empty-state-desc">Tải ảnh lên từ tab bên trái để xem kết quả phân loại độ tươi</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tips-card">
                    <div class="tips-heading">💡 Mẹo chụp ảnh đẹp</div>
                    <ul class="tips-list">
                        <li>Ánh sáng tự nhiên đầy đủ, tránh bóng đổ</li>
                        <li>Chụp cận cảnh, thịt chiếm phần lớn khung hình</li>
                        <li>Góc chụp trực diện, không nghiêng</li>
                        <li>Nền đơn sắc, tương phản với thịt</li>
                    </ul>
                </div>
                
                <div class="info-box blue">
                    <strong>Lưu ý quan trọng:</strong> Kết quả AI chỉ mang tính tham khảo. Luôn kiểm tra thêm bằng khứu giác và xúc giác trước khi sử dụng thực phẩm.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Chụp ảnh từ camera
    with tab2:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="content-card"><div class="card-heading"><div class="card-heading-dot"></div>Chụp ảnh trực tiếp</div>', unsafe_allow_html=True)
            
            if 'camera_enabled' not in st.session_state:
                st.session_state.camera_enabled = False
            
            if not st.session_state.camera_enabled:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-title">Camera đang tắt</div>
                    <div class="empty-state-desc">Nhấn nút bên dưới để bật camera và chụp ảnh</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📷 Bật Camera", type="primary", key="enable_camera", use_container_width=True):
                    st.session_state.camera_enabled = True
                    st.rerun()
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("⏹ Tắt Camera", key="disable_camera", use_container_width=True):
                        st.session_state.camera_enabled = False
                        st.rerun()
                with col_b:
                    if st.button("🔄 Mới", key="new_photo", use_container_width=True):
                        pass
                
                camera_photo = st.camera_input(
                    "", 
                    help="Nhấn vào nút camera để chụp ảnh",
                    key="camera_input",
                    label_visibility="collapsed"
                )
                
                if camera_photo is not None:
                    camera_image = Image.open(camera_photo)
                    st.markdown('<div style="margin-top: 1.5rem;" class="image-frame">', unsafe_allow_html=True)
                    st.image(camera_image, caption="", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
                    if st.button("🔍 Phân tích độ tươi", type="primary", key="camera_predict", use_container_width=True):
                        analyze_image(model, camera_image, col2)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if not st.session_state.camera_enabled:
                st.markdown('<div class="content-card"><div class="card-heading"><div class="card-heading-dot"></div>Kết quả phân tích</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-title">Bật camera để bắt đầu</div>
                    <div class="empty-state-desc">Sử dụng camera để chụp ảnh thịt và xem kết quả phân tích ngay lập tức</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tips-card">
                    <div class="tips-heading">📸 Hướng dẫn chụp</div>
                    <ul class="tips-list">
                        <li>Đặt thịt trên bề mặt phẳng, sáng màu</li>
                        <li>Giữ thiết bị ổn định khi chụp</li>
                        <li>Đảm bảo đủ ánh sáng tự nhiên</li>
                        <li>Chụp từ góc nhìn trực diện</li>
                    </ul>
                </div>
                
                <div class="info-box amber">
                    <strong>Mẹo:</strong> Chỉ bật camera khi cần để tiết kiệm pin và tài nguyên thiết bị.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif camera_photo is None:
                st.markdown('<div class="content-card"><div class="card-heading"><div class="card-heading-dot"></div>Kết quả phân tích</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-title">Sẵn sàng chụp</div>
                    <div class="empty-state-desc">Nhấn vào biểu tượng camera ở cột bên trái để chụp ảnh</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tips-card">
                    <div class="tips-heading">✅ Camera đã sẵn sàng</div>
                    <ul class="tips-list">
                        <li>Nhấn vào nút camera để chụp</li>
                        <li>Có thể tắt camera sau khi chụp xong</li>
                    </ul>
                </div>
                
                <div class="info-box green">
                    <strong>Sẵn sàng!</strong> Chất lượng ảnh tốt sẽ cho kết quả chính xác hơn.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Modern footer
    st.markdown("""
    <div class="app-footer">
        <div style="max-width: 1400px; margin: 0 auto;">
            <strong>MonFresh</strong> © 2024 — Ứng dụng phân tích độ tươi của thịt bằng trí tuệ nhân tạo<br>
            <span style="opacity: 0.7;">Giúp người tiêu dùng lựa chọn thực phẩm an toàn và chất lượng</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
