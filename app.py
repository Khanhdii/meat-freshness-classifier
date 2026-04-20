import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="MonFresh - Quản lý kho thịt tươi sống",
    page_icon="🥩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS HIỆN ĐẠI (APPLE/STRIPE STYLE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --secondary: #f43f5e;
        --accent: #f59e0b;
        --bg-body: #f8fafc;
        --bg-card: #ffffff;
        --text-main: #1e293b;
        --text-muted: #64748b;
        --border: #e2e8f0;
        --radius: 16px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-body);
        color: var(--text-main);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }

    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2.5rem;
        background: var(--bg-card);
        padding: 1.5rem 2rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }

    .brand-logo {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--primary-dark);
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.5px;
    }

    .brand-icon {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }

    .metric-card {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--primary);
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.1);
    }

    .metric-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-main);
        line-height: 1.2;
    }

    .metric-trend {
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
    }

    .trend-up { background: #dcfce7; color: #166534; }
    .trend-down { background: #fee2e2; color: #991b1b; }

    h2, h3 {
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }

    div[data-testid="stDataFrame"] {
        border: none !important;
        border-radius: var(--radius);
        overflow: hidden;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    table {
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
    }
    
    th {
        background-color: #f1f5f9 !important;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        padding: 1rem !important;
        border-bottom: 1px solid var(--border);
    }
    
    td {
        padding: 1rem !important;
        border-bottom: 1px solid var(--border);
        color: var(--text-main);
        font-size: 0.9rem;
    }

    tr:last-child td {
        border-bottom: none;
    }

    tr:hover td {
        background-color: #f8fafc;
    }

    .stButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }

    .stButton > button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }

    .status-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .status-good { background: #dcfce7; color: #166534; }
    .status-warn { background: #fef3c7; color: #92400e; }
    .status-bad { background: #fee2e2; color: #991b1b; }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 1px solid var(--border);
        background-color: #f8fafc;
        padding: 10px 14px;
    }
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }

</style>
""", unsafe_allow_html=True)

# --- KHỞI TẠO DỮ LIỆU MẪU ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame([
        {"Mã SP": "MF001", "Tên sản phẩm": "Thịt Ba Chỉ Heo", "Số lượng (kg)": 45.5, "Ngày nhập": "2023-10-25", "Hạn sử dụng (ngày)": 7, "Trạng thái": "Tốt"},
        {"Mã SP": "MF002", "Tên sản phẩm": "Thịt Bò Mỹ", "Số lượng (kg)": 12.0, "Ngày nhập": "2023-10-24", "Hạn sử dụng (ngày)": 5, "Trạng thái": "Cảnh báo"},
        {"Mã SP": "MF003", "Tên sản phẩm": "Ức Gà Công Nghiệp", "Số lượng (kg)": 80.0, "Ngày nhập": "2023-10-26", "Hạn sử dụng (ngày)": 10, "Trạng thái": "Tốt"},
        {"Mã SP": "MF004", "Tên sản phẩm": "Sườn Non Heo", "Số lượng (kg)": 5.5, "Ngày nhập": "2023-10-20", "Hạn sử dụng (ngày)": 2, "Trạng thái": "Hết hạn"},
        {"Mã SP": "MF005", "Tên sản phẩm": "Cá Hồi Na Uy", "Số lượng (kg)": 15.0, "Ngày nhập": "2023-10-25", "Hạn sử dụng (ngày)": 4, "Trạng thái": "Cảnh báo"},
    ])

df = st.session_state.data

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🍽️ MonFresh Menu")
    st.markdown("---")
    menu = st.radio("Điều hướng", ["Tổng quan", "Quản lý kho", "Nhập hàng", "Báo cáo"], label_visibility="collapsed")
    
    st.markdown("---")
    st.info("💡 **Mẹo:** Cập nhật trạng thái hàng hóa mỗi ngày để đảm bảo chất lượng.")

# --- LOGIC XỬ LÝ ---

def get_status_color(status):
    if status == "Tốt": return "status-good"
    elif status == "Cảnh báo": return "status-warn"
    else: return "status-bad"

def render_dashboard():
    st.markdown(f"""
    <div class="header-container">
        <div class="brand-logo">
            <div class="brand-icon">🥩</div>
            MonFresh System
        </div>
        <div style="text-align: right;">
            <div style="font-weight: 600; color: var(--text-main);">Xin chào, Admin</div>
            <div style="font-size: 0.85rem; color: var(--text-muted);">{datetime.now().strftime('%d/%m/%Y')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    total_stock = df["Số lượng (kg)"].sum()
    warning_count = len(df[df["Trạng thái"] == "Cảnh báo"])
    expired_count = len(df[df["Trạng thái"] == "Hết hạn"])
    total_items = len(df)

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tổng tồn kho</div>
            <div class="metric-value">{total_stock:,.1f} <span style="font-size:1rem; color:var(--text-muted)">kg</span></div>
            <div class="metric-trend trend-up">↗ +12% tuần này</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sản phẩm sắp hết hạn</div>
            <div class="metric-value" style="color: var(--accent)">{warning_count}</div>
            <div class="metric-trend trend-down">Cần xử lý ngay</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Hàng hết hạn</div>
            <div class="metric-value" style="color: var(--secondary)">{expired_count}</div>
            <div class="metric-trend trend-down">Loại bỏ ngay</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tổng mã hàng</div>
            <div class="metric-value">{total_items}</div>
            <div class="metric-trend" style="background:#f1f5f9; color:var(--text-muted)">Đang hoạt động</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📦 Tình trạng kho hiện tại")
    
    df_display = df.copy()
    df_display['Trạng thái'] = df_display['Trạng thái'].apply(lambda x: f'<span class="status-badge {get_status_color(x)}">{x}</span>')
    st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

def render_inventory():
    st.markdown(f"""
    <div class="header-container">
        <div class="brand-logo">
            <div class="brand-icon">📦</div>
            Quản lý chi tiết
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Danh sách tất cả sản phẩm")
    
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        search = st.text_input("🔍 Tìm kiếm sản phẩm...", placeholder="Nhập tên hoặc mã...")
    with col_f2:
        filter_status = st.selectbox("Lọc trạng thái", ["Tất cả", "Tốt", "Cảnh báo", "Hết hạn"])
    
    filtered_df = df.copy()
    if search:
        filtered_df = filtered_df[filtered_df["Tên sản phẩm"].str.contains(search, case=False) | filtered_df["Mã SP"].str.contains(search, case=False)]
    if filter_status != "Tất cả":
        filtered_df = filtered_df[filtered_df["Trạng thái"] == filter_status]
        
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

def render_add_product():
    st.markdown(f"""
    <div class="header-container">
        <div class="brand-logo">
            <div class="brand-icon">➕</div>
            Nhập hàng mới
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("add_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Tên sản phẩm *")
            qty = st.number_input("Số lượng (kg)", min_value=0.1, step=0.1)
            expiry = st.number_input("Hạn sử dụng (ngày)", min_value=1, step=1, value=7)
        with c2:
            code = st.text_input("Mã sản phẩm (Tự động nếu để trống)")
            date_in = st.date_input("Ngày nhập", datetime.now())
            
        submitted = st.form_submit_button("Thêm vào kho", use_container_width=True)
        
        if submitted:
            if not name or qty <= 0:
                st.error("Vui lòng nhập đầy đủ tên và số lượng hợp lệ!")
            else:
                new_code = code if code else f"MF{random.randint(100, 999)}"
                status = "Tốt" if expiry > 5 else ("Cảnh báo" if expiry > 2 else "Hết hạn")
                
                new_row = pd.DataFrame([{
                    "Mã SP": new_code,
                    "Tên sản phẩm": name,
                    "Số lượng (kg)": qty,
                    "Ngày nhập": str(date_in),
                    "Hạn sử dụng (ngày)": expiry,
                    "Trạng thái": status
                }])
                
                st.session_state.data = pd.concat([df, new_row], ignore_index=True)
                st.success(f"Đã thêm thành công sản phẩm **{name}** vào kho MonFresh!")
                st.balloons()

def render_report():
    st.markdown(f"""
    <div class="header-container">
        <div class="brand-logo">
            <div class="brand-icon">📊</div>
            Báo cáo & Phân tích
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Chức năng báo cáo chi tiết đang được cập nhật cho phiên bản MonFresh Pro.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Tỷ lệ hàng hóa theo trạng thái")
        status_counts = df["Trạng thái"].value_counts()
        st.bar_chart(status_counts, color="#10b981")
    
    with c2:
        st.markdown("#### Top 5 sản phẩm tồn kho nhiều nhất")
        top_5 = df.nlargest(5, "Số lượng (kg)")
        st.bar_chart(top_5.set_index("Tên sản phẩm")["Số lượng (kg)"], color="#f59e0b")

# --- MAIN ROUTING ---
if menu == "Tổng quan":
    render_dashboard()
elif menu == "Quản lý kho":
    render_inventory()
elif menu == "Nhập hàng":
    render_add_product()
elif menu == "Báo cáo":
    render_report()
