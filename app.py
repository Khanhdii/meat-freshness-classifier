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
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional color palette with custom theme colors
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Theme Color Variables */
    :root {
        --color-primary: #5D7B6F;      /* Xám xanh đậm - Main color */
        --color-secondary: #A4C3A2;    /* Xanh lá nhạt ấm - Secondary */
        --color-accent: #B0D4B8;       /* Xanh mint nhẹ - Accent */
        --color-neutral: #EAE7D6;      /* Kem be trung tính - Neutral background */
        --color-pastel: #D7F9FA;       /* Xanh ngọc pastel - Pastel accent */
        --color-fresh: #5D7B6F;        /* Fresh status */
        --color-half: #A4C3A2;         /* Half-fresh status */
        --color-spoiled: #D4A5A5;      /* Spoiled status (muted red) */
    }
    
    /* Base styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #FDFCFB 0%, #F5F4F0 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header styling - Elegant gradient with primary color */
    .header-container {
        background: linear-gradient(135deg, var(--color-primary) 0%, #4A6358 100%);
        padding: 1.25rem 2rem;
        margin: -1.5rem -1.5rem 1.5rem -1.5rem;
        position: sticky;
        top: 0;
        z-index: 100;
        border-bottom: 3px solid var(--color-secondary);
    }
    
    .header-content {
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, var(--color-pastel) 0%, var(--color-accent) 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 2px 8px rgba(93, 123, 111, 0.2);
    }
    
    .header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: white;
        line-height: 1.2;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        font-size: 0.8125rem;
        color: var(--color-pastel);
        font-weight: 500;
    }
    
    .header-actions {
        display: flex;
        gap: 0.5rem;
    }
    
    .btn-export {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        color: white;
        cursor: pointer;
        transition: all 0.2s;
        backdrop-filter: blur(10px);
    }
    
    .btn-export:hover {
        background: rgba(255,255,255,0.25);
        border-color: rgba(255,255,255,0.5);
    }
    
    .btn-primary {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.25rem;
        background: linear-gradient(135deg, var(--color-secondary) 0%, var(--color-accent) 100%);
        border: none;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-primary);
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(164, 195, 162, 0.3);
    }
    
    .btn-primary:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(164, 195, 162, 0.4);
    }
    
    /* Stats container - Elegant grid with theme colors */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.25rem;
        margin-bottom: 1.5rem;
    }
    
    .stat-card {
        background: white;
        border: 1px solid var(--color-neutral);
        border-radius: 14px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: var(--color-accent);
    }
    
    .stat-card:hover::before {
        opacity: 1;
    }
    
    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.875rem;
        font-size: 1.4rem;
    }
    
    .stat-icon.blue { 
        background: linear-gradient(135deg, var(--color-pastel) 0%, var(--color-accent) 100%); 
        color: var(--color-primary); 
    }
    .stat-icon.green { 
        background: linear-gradient(135deg, #E8F5E9 0%, var(--color-accent) 100%); 
        color: var(--color-primary); 
    }
    .stat-icon.amber { 
        background: linear-gradient(135deg, #FFF8E1 0%, #FFE0B2 100%); 
        color: #B58900; 
    }
    .stat-icon.red { 
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); 
        color: #C62828; 
    }
    
    .stat-value {
        font-size: 1.625rem;
        font-weight: 700;
        color: var(--color-primary);
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #6B7280;
        font-weight: 500;
    }
    
    .stat-trend {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.625rem;
        border-radius: 9999px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .stat-trend.up { 
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); 
        color: #059669; 
    }
    .stat-trend.down { 
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%); 
        color: #DC2626; 
    }
    
    /* Card styling - Modern with theme colors */
    .card {
        background: white;
        border: 1px solid var(--color-neutral);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: var(--color-accent);
    }
    
    .card-header {
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-neutral);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .card-title {
        font-size: 1.0625rem;
        font-weight: 600;
        color: var(--color-primary);
    }
    
    .card-subtitle {
        font-size: 0.8125rem;
        color: #6B7280;
        margin-top: 0.25rem;
    }
    
    /* Result boxes - Theme colored with gradients */
    .result-box {
        padding: 1.75rem;
        border-radius: 12px;
        border: 2px solid;
        text-align: center;
        background: white;
    }
    
    .result-box.success {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-color: var(--color-fresh);
    }
    
    .result-box.warning {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-color: #D97706;
    }
    
    .result-box.error {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-color: #DC2626;
    }
    
    .result-title {
        font-size: 1.625rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: var(--color-primary);
    }
    
    .result-class-en {
        font-size: 0.9375rem;
        color: #6B7280;
        margin-bottom: 0.75rem;
    }
    
    .result-confidence {
        font-size: 0.9375rem;
        font-weight: 600;
        color: var(--color-primary);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.5rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.8125rem;
        font-weight: 600;
    }
    
    .status-badge.success { 
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%); 
        color: #166534; 
        border: 1px solid #86EFAC; 
    }
    .status-badge.warning { 
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); 
        color: #92400e; 
        border: 1px solid #FCD34D; 
    }
    .status-badge.error { 
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); 
        color: #991b1b; 
        border: 1px solid #FCA5A5; 
    }
    
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
    }
    
    .status-dot.success { 
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); 
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.4);
    }
    .status-dot.warning { 
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
    }
    .status-dot.error { 
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
    }
    
    /* Progress bars - Theme colored */
    .progress-item {
        margin-bottom: 1.125rem;
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    
    .progress-label-name {
        font-weight: 600;
        color: var(--color-primary);
    }
    
    .progress-label-value {
        color: #6B7280;
        font-weight: 500;
    }
    
    .progress-track {
        background: var(--color-neutral);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .progress-fill.fresh {
        background: linear-gradient(90deg, var(--color-fresh) 0%, var(--color-accent) 100%);
    }
    .progress-fill.half {
        background: linear-gradient(90deg, #D97706 0%, #FBBF24 100%);
    }
    .progress-fill.spoiled {
        background: linear-gradient(90deg, #DC2626 0%, #EF4444 100%);
    }
    
    /* Info/recommendation boxes - Theme styled */
    .success-box, .warning-box, .error-box {
        padding: 1.125rem;
        border-radius: 10px;
        margin-top: 1rem;
        line-height: 1.7;
        font-size: 0.875rem;
        border-left: 4px solid;
    }
    
    .success-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-left-color: var(--color-fresh);
        color: #166534;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left-color: #D97706;
        color: #92400e;
    }
    
    .error-box {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left-color: #DC2626;
        color: #991b1b;
    }
    
    /* Sidebar styling - Theme integrated */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FDFCFB 0%, #F5F4F0 100%);
        border-right: 1px solid var(--color-neutral);
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--color-primary);
        text-transform: uppercase;
        letter-spacing: 0.075em;
        margin-bottom: 0.875rem;
        padding-bottom: 0.625rem;
        border-bottom: 2px solid var(--color-neutral);
    }
    
    .info-box {
        background: linear-gradient(135deg, var(--color-pastel) 0%, #F0FDFA 100%);
        border: 1px solid var(--color-accent);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 0.8125rem;
        color: var(--color-primary);
    }
    
    .class-item {
        display: flex;
        align-items: flex-start;
        gap: 0.875rem;
        padding: 0.875rem;
        background: white;
        border-radius: 10px;
        margin-bottom: 0.625rem;
        border: 1px solid var(--color-neutral);
        transition: all 0.2s ease;
    }
    
    .class-item:hover {
        border-color: var(--color-accent);
        background: var(--color-pastel);
    }
    
    .class-indicator {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.8125rem;
        flex-shrink: 0;
    }
    
    .class-indicator.fresh {
        background: linear-gradient(135deg, var(--color-fresh) 0%, var(--color-accent) 100%);
    }
    .class-indicator.half {
        background: linear-gradient(135deg, #D97706 0%, #FBBF24 100%);
    }
    .class-indicator.spoiled {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
    }
    
    .class-info-title {
        font-weight: 600;
        color: var(--color-primary);
        font-size: 0.875rem;
    }
    
    .class-info-desc {
        font-size: 0.75rem;
        color: #6B7280;
    }
    
    /* Tips section - Theme styled */
    .tips-section {
        background: linear-gradient(135deg, white 0%, var(--color-pastel) 100%);
        padding: 1.125rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid var(--color-accent);
    }
    
    .tips-title {
        font-weight: 700;
        color: var(--color-primary);
        margin-bottom: 0.625rem;
        font-size: 0.875rem;
    }
    
    .tips-list {
        margin: 0;
        padding-left: 1.25rem;
        color: #4B5563;
        font-size: 0.8125rem;
        line-height: 1.875;
    }
    
    .tips-list li {
        margin-bottom: 0.375rem;
    }
    
    .note-box {
        padding: 1.125rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 0.8125rem;
        border: 1px solid;
    }
    
    .note-box.info { 
        background: linear-gradient(135deg, var(--color-pastel) 0%, #E0F2FE 100%); 
        border-color: var(--color-accent); 
        color: #0369a1; 
    }
    .note-box.warning { 
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); 
        border-color: #FCD34D; 
        color: #92400e; 
    }
    .note-box.success { 
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); 
        border-color: #86EFAC; 
        color: #166534; 
    }
    
    /* Empty state - Theme styled */
    .empty-state {
        text-align: center;
        padding: 2.5rem 1.5rem;
        color: #6B7280;
        background: linear-gradient(135deg, white 0%, var(--color-pastel) 100%);
        border-radius: 14px;
        border: 2px dashed var(--color-accent);
    }
    
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.6;
    }
    
    .empty-state-title {
        font-weight: 700;
        color: var(--color-primary);
        margin-bottom: 0.625rem;
        font-size: 1rem;
    }
    
    .empty-state-desc {
        font-size: 0.875rem;
        color: #6B7280;
    }
    
    /* File uploader styling - Theme integrated */
    .stFileUploader {
        margin-bottom: 0.625rem;
    }
    
    .stFileUploader > div {
        border: 2px dashed var(--color-accent);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, white 0%, var(--color-pastel) 100%);
    }
    
    .stFileUploader > div:hover {
        border-color: var(--color-primary);
        background: linear-gradient(135deg, var(--color-pastel) 0%, #E0F2FE 100%);
        transform: scale(1.01);
    }
    
    /* Button styling - Theme colored */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.875rem;
        padding: 0.625rem 1.25rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button[type="primary"] {
        background: linear-gradient(135deg, var(--color-primary) 0%, #4A6358 100%);
        color: white;
        border: none;
        box-shadow: 0 2px 8px rgba(93, 123, 111, 0.3);
    }
    
    .stButton > button[type="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(93, 123, 111, 0.4);
    }
    
    /* Tabs styling - Block style with theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.625rem;
        background: var(--color-neutral);
        padding: 0.375rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        color: #6B7280;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: var(--color-primary);
        box-shadow: 0 2px 8px rgba(93, 123, 111, 0.15);
    }
    
    /* Image container - Theme styled */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stImage img {
        border-radius: 12px;
    }
    
    .image-wrapper {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid var(--color-neutral);
        background: white;
        transition: all 0.3s ease;
    }
    
    .image-wrapper:hover {
        border-color: var(--color-accent);
    }
    
    /* Alert/Info messages - Theme styled */
    .stAlert {
        border-radius: 10px;
        font-size: 0.875rem;
    }
    
    /* Remove default Streamlit padding issues */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Column spacing */
    .element-container {
        margin-bottom: 0.875rem;
    }
    
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Upload zone - Theme styled */
    .upload-zone {
        border: 2px dashed var(--color-accent);
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, white 0%, var(--color-pastel) 100%);
    }
    
    .upload-zone:hover {
        border-color: var(--color-primary);
        background: linear-gradient(135deg, var(--color-pastel) 0%, #E0F2FE 100%);
        transform: scale(1.02);
    }
    
    .upload-icon {
        width: 72px;
        height: 72px;
        background: linear-gradient(135deg, var(--color-neutral) 0%, #E5E5E5 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.25rem;
        font-size: 1.75rem;
        color: var(--color-primary);
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover .upload-icon {
        background: linear-gradient(135deg, var(--color-accent) 0%, var(--color-secondary) 100%);
        color: white;
        transform: rotate(-5deg);
    }
    
    .upload-title {
        font-size: 0.9375rem;
        font-weight: 700;
        color: var(--color-primary);
    }
    
    .upload-desc {
        font-size: 0.8125rem;
        color: #6B7280;
        margin-top: 0.625rem;
    }
    
    /* Footer - Theme styled */
    .footer {
        background: linear-gradient(135deg, var(--color-primary) 0%, #4A6358 100%);
        border-top: 3px solid var(--color-secondary);
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        color: white;
    }
    
    .footer-brand {
        font-size: 1.25rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }
    
    .footer-text {
        font-size: 0.875rem;
        color: var(--color-pastel);
        line-height: 1.6;
    }
    
    .footer-disclaimer {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
        margin-top: 1rem;
        font-style: italic;
    }
    
    /* Metric cards enhancement */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid var(--color-neutral);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--color-accent);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.8125rem;
        color: #6B7280;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--color-primary);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--color-neutral);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
        border-radius: 4px;
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
    """Phân tích ảnh và hiển thị kết quả theo phong cách thương mại"""
    with st.spinner(""):
        try:
            predicted_class, confidence, all_predictions = predict_image(model, image)
            
            # Hiển thị kết quả trong cột được chỉ định
            with result_col:
                class_name = CLASS_NAMES[predicted_class]
                class_name_vi = CLASS_NAMES_VI[predicted_class]
                color_class = ['success', 'warning', 'error'][predicted_class]
                
                # Status badge
                status_badge = f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span class="status-badge {color_class}">
                        <span class="status-dot {color_class}"></span>
                        {class_name_vi}
                    </span>
                    <span style="font-size: 0.75rem; color: #94a3b8;">Sample ID: #MF-2024-{np.random.randint(1000, 9999)}</span>
                </div>
                """
                st.markdown(status_badge, unsafe_allow_html=True)
                
                # Result box với styling đẹp
                st.markdown(f"""
                <div class="result-box {color_class}">
                    <div class="result-title" style="color: {CLASS_COLORS[predicted_class]}">{class_name_vi}</div>
                    <div class="result-class-en">{class_name}</div>
                    <div class="result-confidence">Độ tin cậy: {confidence:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Biểu đồ chi tiết các xác suất
                st.markdown('<div class="card" style="margin-top: 1.5rem;"><div class="card-header"><div class="card-title">Chi tiết xác suất</div></div>', unsafe_allow_html=True)
                
                for i, (class_id, prob) in enumerate(zip(CLASS_NAMES.keys(), all_predictions)):
                    cn = CLASS_NAMES[class_id]
                    cn_vi = CLASS_NAMES_VI[class_id]
                    bar_color = CLASS_COLORS[class_id]
                    
                    # Custom progress bar với màu sắc
                    st.markdown(f"""
                    <div class="progress-item">
                        <div class="progress-label">
                            <span class="progress-label-name">{cn_vi} ({cn})</span>
                            <span class="progress-label-value">{prob:.2%}</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" style="background: {bar_color}; width: {prob*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Khuyến nghị
                st.markdown('<div class="card"><div class="card-header"><div class="card-title">Khuyến nghị</div></div>', unsafe_allow_html=True)
                
                if predicted_class == 0:
                    st.markdown("""
                    <div class="success-box">
                        <strong>Thịt còn tươi</strong><br>
                        Sản phẩm ở trạng thái tốt nhất, có thể sử dụng an toàn ngay lập tức.
                        Nên bảo quản ở nhiệt độ thích hợp để duy trì độ tươi.
                    </div>
                    """, unsafe_allow_html=True)
                elif predicted_class == 1:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Thịt bán tươi</strong><br>
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
        <div style="margin-bottom: 1.5rem;">
            <div class="sidebar-title">Thông tin hệ thống</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong style="color: #0f172a;">Kích thước đầu vào:</strong><br>
            <span style="color: #64748b;">{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]} pixels</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
            <div class="sidebar-title">Các lớp phân loại</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-indicator" style="background: #22c55e;">F</div>
            <div>
                <div class="class-info-title">Tươi (Fresh)</div>
                <div class="class-info-desc">Sản phẩm chất lượng tốt nhất</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-indicator" style="background: #f59e0b;">H</div>
            <div>
                <div class="class-info-title">Bán tươi (Half)</div>
                <div class="class-info-desc">Cần sử dụng sớm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="class-item">
            <div class="class-indicator" style="background: #ef4444;">S</div>
            <div>
                <div class="class-info-title">Hỏng (Spoiled)</div>
                <div class="class-info-desc">Không nên sử dụng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
            <div class="sidebar-title">Hướng dẫn</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="font-size: 0.8125rem; line-height: 1.75; color: #475569;">
            <strong style="color: #0f172a;">Upload:</strong><br>
            • Chọn ảnh từ thiết bị<br>
            • Click "Phân tích độ tươi"<br>
            • Xem kết quả chi tiết<br><br>
            
            <strong style="color: #0f172a;">Camera:</strong><br>
            • Bật camera để kích hoạt<br>
            • Chụp ảnh thịt cần phân loại<br>
            • Nhận kết quả ngay lập tức<br><br>
            
            <em style="color: #94a3b8;">Camera chỉ bật khi cần để tiết kiệm tài nguyên</em>
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
                <div class="upload-icon">📁</div>
                <div class="upload-title">Click để tải ảnh lên hoặc kéo thả</div>
                <div class="upload-desc">Hỗ trợ PNG, JPG, JPEG (tối đa 800x400px)</div>
            </div>
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
                    <div class="empty-state-icon">📊</div>
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
                    <div class="empty-state-icon">📷</div>
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
                    <div class="empty-state-icon">📊</div>
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
                    <div class="empty-state-icon">📸</div>
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
