import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# ============================================================================
# CẤU HÌNH TRANG & CUSTOM CSS - UNIQUE VIBRANT DESIGN
# ============================================================================

st.set_page_config(
    page_title="MonFresh - AI Meat Freshness Analysis",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - MODERN SAAS PROFESSIONAL DESIGN
CUSTOM_CSS = """
<style>
/* Import Google Fonts - Premium SaaS Look */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset */
* {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

/* MODERN PROFESSIONAL COLOR PALETTE */
:root {
    --primary-teal: #0D9488;
    --primary-teal-dark: #0F766E;
    --primary-teal-light: #14B8A6;
    --secondary-blue: #0EA5E9;
    --accent-coral: #F43F5E;
    --success-green: #10B981;
    --warning-amber: #F59E0B;
    --error-red: #EF4444;
    --slate-50: #F8FAFC;
    --slate-100: #F1F5F9;
    --slate-200: #E2E8F0;
    --slate-300: #CBD5E1;
    --slate-600: #475569;
    --slate-700: #334155;
    --slate-800: #1E293B;
    --slate-900: #0F172A;
}

/* Main App Background - Clean Professional Gradient */
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 50%, #E2E8F0 100%);
    background-attachment: fixed;
}

/* Header - Modern Gradient with Professional Look */
.header-container {
    background: linear-gradient(135deg, var(--primary-teal) 0%, var(--secondary-blue) 100%);
    padding: 3rem 0;
    position: relative;
    overflow: hidden;
    border-radius: 0 0 32px 32px;
    box-shadow: 0 20px 40px rgba(13, 148, 136, 0.15);
}

.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-20px); }
}

.header-content {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 1;
}

.header-logo {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.logo-icon {
    width: 60px;
    height: 60px;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.header-title {
    color: white;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.header-subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 0.95rem;
    font-weight: 500;
}

/* Stats Bar - Modern Clean Cards */
.stats-container {
    max-width: 1400px;
    margin: -2rem auto 0;
    padding: 0 2.5rem;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.5rem;
    position: relative;
    z-index: 2;
}

.stat-card {
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 20px;
    padding: 1.75rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 10px 15px -3px rgba(0, 0, 0, 0.03);
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-teal) 0%, var(--secondary-blue) 100%);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px -8px rgba(0, 0, 0, 0.1);
}

.stat-card:hover::before {
    transform: scaleX(1);
}

.stat-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 700;
    color: var(--slate-900);
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--primary-teal) 0%, var(--secondary-blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    color: var(--slate-600);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Main Content */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 3.5rem 2.5rem;
}

/* Cards - Modern Clean Design */
.card {
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 10px 15px -3px rgba(0, 0, 0, 0.03);
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 40px -8px rgba(0, 0, 0, 0.1);
}

.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--slate-200);
}

.card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--slate-900);
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.card-badge {
    background: linear-gradient(135deg, var(--primary-teal) 0%, var(--primary-teal-light) 100%);
    color: white;
    padding: 0.5rem 1.25rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.075em;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
}

/* Result Box - Modern Professional */
.result-box {
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 24px;
    padding: 3rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 10px 15px -3px rgba(0, 0, 0, 0.03);
}

.result-box::after {
    content: '';
    position: absolute;
    top: 8px;
    left: 8px;
    right: 8px;
    bottom: 8px;
    border-radius: 18px;
    border: 2px dashed var(--slate-300);
    opacity: 0.3;
}

.result-box.success {
    background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
    border-color: var(--success-green);
}
.result-box.success .result-title { 
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-box.warning {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border-color: var(--warning-amber);
}
.result-box.warning .result-title { 
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-box.error {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    border-color: var(--error-red);
}
.result-box.error .result-title { 
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--slate-600);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1rem;
}

.result-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
    letter-spacing: -0.03em;
    color: var(--slate-900);
}

.result-confidence {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--slate-700);
    font-weight: 600;
    background: white;
    padding: 0.6rem 1.5rem;
    border-radius: 12px;
    display: inline-block;
    margin-top: 1.25rem;
    border: 1px solid var(--slate-200);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Probability Bars - Modern Clean Style */
.probability-item {
    margin-bottom: 1.5rem;
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 16px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}

.probability-item:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}

.probability-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.probability-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    color: var(--slate-700);
    font-size: 0.9rem;
}

.probability-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    color: var(--primary-teal);
    font-size: 1rem;
}

.probability-bar-bg {
    background: var(--slate-100);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}

.probability-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    background: linear-gradient(90deg, var(--primary-teal) 0%, var(--secondary-blue) 100%);
    box-shadow: 0 2px 8px rgba(13, 148, 136, 0.3);
}

/* Info Boxes - Modern Clean Design */
.info-box, .success-box, .warning-box, .error-box {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1.25rem 0;
    border: none;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.info-box {
    background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
    color: var(--slate-700);
    border-left: 4px solid var(--secondary-blue);
}

.success-box {
    background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
    color: var(--slate-700);
    border-left: 4px solid var(--success-green);
}

.warning-box {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    color: var(--slate-700);
    border-left: 4px solid var(--warning-amber);
}

.error-box {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    color: var(--slate-700);
    border-left: 4px solid var(--error-red);
}

/* Progress Bar - Modern Gradient */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary-teal) 0%, var(--secondary-blue) 100%);
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(13, 148, 136, 0.3);
}

/* Buttons - Modern Professional Style */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-teal) 0%, var(--primary-teal-light) 100%);
    color: white;
    border: none;
    padding: 1rem 2.5rem;
    border-radius: 12px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(13, 148, 136, 0.4);
}

/* Sidebar - Modern Clean Design */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    border-right: 1px solid var(--slate-200);
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: var(--slate-700) !important;
}

.sidebar-section {
    margin: 2rem 0;
}

.sidebar-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--slate-600);
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--slate-200);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.class-item {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding: 1.25rem;
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 16px;
    margin: 1rem 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.class-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    background: var(--slate-50);
}

.class-indicator {
    width: 3.5rem;
    height: 3.5rem;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    color: white;
}

.class-info-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    color: var(--slate-900);
    font-size: 0.9rem;
}

.class-info-desc {
    font-size: 0.8rem;
    color: var(--slate-600);
    margin-top: 0.35rem;
    line-height: 1.5;
}

/* Tabs - Modern Clean Style */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    justify-content: center;
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 16px;
    padding: 0.75rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.stTabs [data-baseweb="tab"] {
    padding: 0.875rem 2rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    color: var(--slate-600);
    border: none;
    border-radius: 12px;
    background: transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--primary-teal);
    background: var(--slate-50);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary-teal) 0%, var(--primary-teal-light) 100%);
    color: white;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
}

/* File Uploader - Modern Clean */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--slate-300);
    border-radius: 20px;
    padding: 3rem;
    background: white;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary-teal);
    background: var(--slate-50);
    transform: scale(1.01);
}

/* Empty State - Modern Professional */
.empty-state {
    text-align: center;
    padding: 4rem;
    background: white;
    border-radius: 24px;
    border: 2px dashed var(--slate-300);
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1.5rem;
}

.empty-state-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--slate-900);
    margin-bottom: 0.75rem;
}

.empty-state-desc {
    color: var(--slate-600);
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Tips Section - Modern Card */
.tips-section {
    background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-top: 2rem;
    border-left: 4px solid var(--secondary-blue);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}

.tips-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    color: var(--slate-700);
    margin-bottom: 1.25rem;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.tips-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.tips-list li {
    padding: 0.75rem 0;
    color: var(--slate-700);
    font-size: 0.95rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    line-height: 1.6;
    font-weight: 500;
}

.tips-list li:before {
    content: "✓";
    color: var(--primary-teal);
    font-weight: 700;
    font-size: 1.1rem;
}

/* Footer - Modern Professional */
.footer {
    background: linear-gradient(135deg, var(--slate-800) 0%, var(--slate-900) 100%);
    color: white;
    padding: 3.5rem 0;
    margin-top: 5rem;
    text-align: center;
    border-top: 4px solid var(--primary-teal);
}

.footer-content {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 2.5rem;
}

.footer-brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-teal-light) 0%, var(--secondary-blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}

.footer-text {
    font-size: 1rem;
    line-height: 1.8;
    margin-bottom: 2rem;
    color: rgba(255, 255, 255, 0.7);
}

.footer-disclaimer {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.4);
    padding-top: 2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
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
    gap: 2rem;
}

/* Image Styling - Modern Borders */
.stImage img {
    border-radius: 20px;
    border: 1px solid var(--slate-200);
    box-shadow: 0 12px 32px rgba(0,0,0,0.08);
}

/* Divider - Modern Style */
hr {
    border-color: var(--slate-200);
    border-style: solid;
    border-width: 1px 0 0 0;
    margin: 2.5rem 0;
}

/* Metric Cards - Modern Clean */
.metric-card {
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 32px rgba(0,0,0,0.08);
}

.metric-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-teal) 0%, var(--secondary-blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.35rem;
}

.metric-label {
    color: var(--slate-600);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Feature Grid - Modern Layout */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin: 2.5rem 0;
}

.feature-card {
    background: white;
    border: 1px solid var(--slate-200);
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-teal) 0%, var(--secondary-blue) 100%);
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}

.feature-icon {
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, var(--primary-teal) 0%, var(--primary-teal-light) 100%);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
    color: white;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(13, 148, 136, 0.3);
}

.feature-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    color: var(--slate-900);
    font-size: 1.15rem;
    margin-bottom: 0.75rem;
}

.feature-desc {
    color: var(--slate-600);
    font-size: 0.9rem;
    line-height: 1.6;
    opacity: 0.85;
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
