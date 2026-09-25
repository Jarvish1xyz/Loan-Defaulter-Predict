"""
FinSecure AI - Loan Default Risk Intelligence & Prediction Dashboard
Dark Mode Edition | Built with Streamlit, Scikit-learn, and Plotly.
"""

import os
import json
import pickle as pkl
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSecure AI | Loan Default Risk Intelligence (Dark Mode)",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS - PREMIUM MODERN DARK MODE PALETTE
# -----------------------------------------------------------------------------
DARK_CUSTOM_CSS = """
<style>
    /* Global Typography & Base Dark Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #F8FAFC;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }
    
    /* Top Header Banner */
    .hero-banner-dark {
        background: linear-gradient(135deg, #0F172A 0%, #0369A1 60%, #0284C7 100%);
        color: #FFFFFF;
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px -5px rgba(2, 132, 199, 0.35), 0 0 15px rgba(56, 189, 248, 0.15);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    .hero-banner-dark::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, rgba(56, 189, 248, 0) 70%);
        border-radius: 50%;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.4rem;
        color: #FFFFFF;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #BAE6FD;
        max-width: 820px;
        line-height: 1.5;
    }
    
    .badge-pill-dark {
        display: inline-block;
        background: rgba(56, 189, 248, 0.18);
        backdrop-filter: blur(10px);
        color: #38BDF8;
        padding: 5px 15px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    
    /* Dark Glass Card Container */
    .glass-card-dark {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card-dark:hover {
        border-color: #0284C7;
        box-shadow: 0 8px 25px -4px rgba(2, 132, 199, 0.25);
    }

    .card-header-dark {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #1F2937;
    }
    
    /* KPI Ribbon */
    .kpi-card-dark {
        background: #111827;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        border: 1px solid #1F2937;
        border-left: 5px solid #0284C7;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .kpi-title-dark {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.04em;
    }
    
    .kpi-value-dark {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 0.2rem;
    }
    
    .kpi-sub-dark {
        font-size: 0.78rem;
        color: #38BDF8;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    
    /* Status Badges */
    .verdict-box-safe-dark {
        background: linear-gradient(135deg, #064E3B 0%, #022C22 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        color: #6EE7B7;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.25);
    }
    
    .verdict-box-danger-dark {
        background: linear-gradient(135deg, #7F1D1D 0%, #450A0A 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        color: #FCA5A5;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.25);
    }
    
    .verdict-title-dark {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: #FFFFFF;
    }
    
    .verdict-desc-dark {
        font-size: 0.95rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Custom Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.4);
        transition: all 0.2s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.5);
        border-color: #38BDF8;
        transform: translateY(-1px);
    }
    
    /* Dark Inputs & Controls */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #F8FAFC !important;
    }

    /* Sidebar Dark Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid #1F2937;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
        color: #94A3B8;
        padding: 8px 18px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        border: 1px solid #334155;
    }

    /* Sidebar navigation */
    .sidebar-nav-label {
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0.25rem 0 0.55rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        display: none;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.25rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        border: 1px solid transparent;
        border-radius: 10px;
        color: #94A3B8;
        cursor: pointer;
        font-weight: 600;
        padding: 0.55rem 0.65rem;
        transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #1E293B;
        color: #E0F2FE;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(2, 132, 199, 0.25), rgba(2, 132, 199, 0.05));
        border-color: rgba(56, 189, 248, 0.45);
        color: #38BDF8;
        box-shadow: inset 3px 0 0 #38BDF8;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {
        display: none;
    }

    /* Performance ranking cards */
    .ranking-card {
        background: linear-gradient(135deg, #111827 0%, #0F1B2D 100%);
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 1rem;
        margin: 0.55rem 0;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.2);
    }

    .ranking-card:first-of-type {
        border-color: #0EA5E9;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.16);
    }

    .ranking-head, .ranking-metrics {
        display: grid;
        grid-template-columns: 44px minmax(140px, 1.3fr) repeat(8, minmax(65px, 1fr));
        gap: 0.45rem;
        align-items: center;
    }

    .ranking-head {
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0 1rem 0.45rem;
    }

    .rank-badge {
        align-items: center;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 9px;
        color: #38BDF8;
        display: flex;
        font-size: 1rem;
        font-weight: 800;
        height: 34px;
        justify-content: center;
        width: 34px;
    }

    .ranking-model {
        color: #F8FAFC;
        font-size: 0.94rem;
        font-weight: 700;
    }

    .ranking-score {
        color: #38BDF8;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .ranking-value {
        color: #CBD5E1;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .ranking-value span {
        color: #64748B;
        display: block;
        font-size: 0.64rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        text-transform: uppercase;
    }

    .dataset-stat {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 12px;
        padding: 1rem;
    }

    .dataset-stat-label {
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .dataset-stat-value {
        color: #F8FAFC;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .batch-row {
        background: #111827;
        border: 1px solid #263449;
        border-left: 4px solid #0284C7;
        border-radius: 12px;
        margin: 0.55rem 0;
        padding: 0.85rem 1rem;
    }

    .batch-row-head {
        align-items: center;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.65rem;
    }

    .batch-row-id {
        color: #F8FAFC;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .batch-risk {
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .batch-models {
        display: grid;
        gap: 0.45rem;
        grid-template-columns: repeat(5, 1fr);
    }

    .batch-model {
        background: #0F172A;
        border-radius: 8px;
        padding: 0.45rem;
        text-align: center;
    }

    .batch-model-name {
        color: #64748B;
        display: block;
        font-size: 0.62rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .batch-model-value {
        color: #CBD5E1;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-weight: 800;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }
</style>
"""
st.markdown(DARK_CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA & MODEL LOADING (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    """Load serialized models, preprocessor, and metadata."""
    models_dir = "models"
    lr_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    dt_path = os.path.join(models_dir, "decision_tree_model.pkl")
    knn_path = os.path.join(models_dir, "knn_model.pkl")
    bagging_path = os.path.join(models_dir, "bagging_model.pkl")
    boosting_path = os.path.join(models_dir, "boosting_model.pkl")
    pre_path = os.path.join(models_dir, "preprocessor.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    required_paths = [lr_path, dt_path, knn_path, bagging_path, boosting_path, pre_path, meta_path]
    if not all(os.path.exists(p) for p in required_paths):
        return None, None, None, None, None, None, None

    def _load_pkl(file_path):
        with open(file_path, "rb") as f:
            return pkl.load(f)

    lr_model = _load_pkl(lr_path)
    dt_model = _load_pkl(dt_path)
    knn_model = _load_pkl(knn_path)
    bagging_model = _load_pkl(bagging_path)
    boosting_model = _load_pkl(boosting_path)
    preprocessor = _load_pkl(pre_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return lr_model, dt_model, knn_model, bagging_model, boosting_model, preprocessor, metadata

@st.cache_data
def load_sample_test_data():
    sample_path = "models/sample_test_data.csv"
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    return None

lr_model, dt_model, knn_model, bagging_model, boosting_model, preprocessor, metadata = load_ml_artifacts()

if lr_model is None:
    st.error("⚠️ Trained models not found in `models/` directory! Please run `python train_model.py` first.")
    st.stop()

# Extract per-model tuned decision thresholds
_thresholds = metadata.get("thresholds", {}) or {}
LR_TH = _thresholds.get("logistic_regression", 0.50)
DT_TH = _thresholds.get("decision_tree", 0.50)
KNN_TH = _thresholds.get("knn", 0.50)
BAGGING_TH = _thresholds.get("bagging", 0.50)
BOOSTING_TH = _thresholds.get("boosting", 0.50)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def predict_single_applicant(model, preprocessor, input_dict, threshold=0.5):
    """Transform input dict, predict default probability and class label using the tuned decision threshold."""
    df_input = pd.DataFrame([input_dict])
    processed_features = preprocessor.transform(df_input)
    prob_default = float(model.predict_proba(processed_features)[0, 1])
    pred_label = int(prob_default >= threshold)
    return prob_default, pred_label

def calculate_emi(loan_amount, annual_interest_rate, term_months):
    """Calculate monthly loan installment (EMI)."""
    if term_months <= 0 or loan_amount <= 0:
        return 0.0
    monthly_rate = (annual_interest_rate / 100) / 12
    if monthly_rate == 0:
        return loan_amount / term_months
    emi = (loan_amount * monthly_rate * ((1 + monthly_rate) ** term_months)) / (((1 + monthly_rate) ** term_months) - 1)
    return emi

def render_risk_gauge_dark(risk_score_percent):
    """Render a modern dark-themed Plotly circular Gauge / Risk Meter."""
    if risk_score_percent < 35:
        bar_color = "#10B981"  # Emerald
        status_text = "LOW RISK"
    elif risk_score_percent < 65:
        bar_color = "#F59E0B"  # Amber
        status_text = "MODERATE RISK"
    else:
        bar_color = "#EF4444"  # Crimson
        status_text = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Risk Level: {status_text}</b>", 'font': {'size': 18, 'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'}},
        delta={'reference': 50, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
        number={'suffix': "%", 'font': {'size': 40, 'color': bar_color, 'weight': 800}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#64748B", 'tickfont': {'size': 12, 'color': '#94A3B8'}},
            'bar': {'color': bar_color, 'thickness': 0.38},
            'bgcolor': "#1E293B",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.20)'},
                {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.20)'},
                {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.20)'}
            ],
            'threshold': {
                'line': {'color': "#F87171", 'width': 4},
                'thickness': 0.8,
                'value': 65
            }
        }
    ))

    fig.update_layout(
        height=290,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#F8FAFC")
    )
    return fig

# -----------------------------------------------------------------------------
# SIDEBAR - CONFIGURATION & PRESETS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank-building.png", width=65)
    st.markdown("## FinSecure AI")
    st.caption("Credit risk intelligence workspace")

    st.markdown('<div class="sidebar-nav-label">Workspace</div>', unsafe_allow_html=True)
    nav_items = [
        "🎯 Loan Default Predictor",
        "📊 Model Performance & Accuracy",
        "📁 Batch CSV Risk Analyzer",
        "📈 Dataset & EDA Insights",
        "ℹ️ Architecture & Documentation"
    ]
    selected_page = st.radio(
        "Application navigation",
        nav_items,
        horizontal=False,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ **Model Configuration**")
    
    selected_model_name = st.selectbox(
        "🧠 **Select Classification Model**",
        ["Logistic Regression", "Decision Tree", "KNN", "Bagging", "Boosting", "Ensemble Consensus (All Models)"],
        index=0,
        help="Compare classic and ensemble learning approaches including KNN, Bagging, and Boosting."
    )

    st.markdown("---")
    st.markdown("### ⚡ **Quick-Fill Borrower Presets**")
    preset_choice = st.selectbox(
        "Load applicant preset profile:",
        [
            "Custom Profile",
            "🌟 Prime Low-Risk Borrower",
            "⚖️ Moderate-Risk Mid-Income",
            "⚠️ High-Risk Overleveraged Applicant",
            "🎓 High-Debt Young Graduate"
        ]
    )

    # Preset dictionaries
    presets = {
        "🌟 Prime Low-Risk Borrower": {
            "Age": 52, "Income": 135000, "LoanAmount": 45000, "CreditScore": 790,
            "MonthsEmployed": 95, "NumCreditLines": 2, "InterestRate": 6.5,
            "LoanTerm": 36, "DTIRatio": 0.22, "Education": "Master's",
            "EmploymentType": "Full-time", "MaritalStatus": "Married",
            "HasMortgage": "Yes", "HasDependents": "Yes", "LoanPurpose": "Home",
            "HasCoSigner": "Yes"
        },
        "⚖️ Moderate-Risk Mid-Income": {
            "Age": 38, "Income": 68000, "LoanAmount": 95000, "CreditScore": 610,
            "MonthsEmployed": 45, "NumCreditLines": 3, "InterestRate": 12.8,
            "LoanTerm": 48, "DTIRatio": 0.48, "Education": "Bachelor's",
            "EmploymentType": "Full-time", "MaritalStatus": "Single",
            "HasMortgage": "No", "HasDependents": "No", "LoanPurpose": "Auto",
            "HasCoSigner": "No"
        },
        "⚠️ High-Risk Overleveraged Applicant": {
            "Age": 24, "Income": 28000, "LoanAmount": 180000, "CreditScore": 460,
            "MonthsEmployed": 8, "NumCreditLines": 4, "InterestRate": 21.5,
            "LoanTerm": 60, "DTIRatio": 0.82, "Education": "High School",
            "EmploymentType": "Unemployed", "MaritalStatus": "Single",
            "HasMortgage": "No", "HasDependents": "Yes", "LoanPurpose": "Business",
            "HasCoSigner": "No"
        },
        "🎓 High-Debt Young Graduate": {
            "Age": 26, "Income": 42000, "LoanAmount": 85000, "CreditScore": 540,
            "MonthsEmployed": 14, "NumCreditLines": 3, "InterestRate": 16.0,
            "LoanTerm": 60, "DTIRatio": 0.65, "Education": "Bachelor's",
            "EmploymentType": "Part-time", "MaritalStatus": "Single",
            "HasMortgage": "No", "HasDependents": "No", "LoanPurpose": "Education",
            "HasCoSigner": "No"
        }
    }

    current_vals = presets.get(preset_choice, None)

    st.markdown("---")
    st.markdown("### 🛡️ **Model Accuracy Benchmark**")
    lr_acc = metadata["metrics"]["logistic_regression"]["accuracy"]
    dt_acc = metadata["metrics"]["decision_tree"]["accuracy"]
    knn_acc = metadata["metrics"]["knn"]["accuracy"]
    bagging_acc = metadata["metrics"]["bagging"]["accuracy"]
    boosting_acc = metadata["metrics"]["boosting"]["accuracy"]
    lr_auc = metadata["metrics"]["logistic_regression"]["roc_auc"]
    dt_auc = metadata["metrics"]["decision_tree"]["roc_auc"]
    knn_auc = metadata["metrics"]["knn"]["roc_auc"]
    bagging_auc = metadata["metrics"]["bagging"]["roc_auc"]
    boosting_auc = metadata["metrics"]["boosting"]["roc_auc"]

    st.markdown(f"""
    <div style="background:#111827; padding:14px; border-radius:12px; border-left:4px solid #38BDF8; font-size:0.85rem; border:1px solid #1E293B;">
        <span style="color:#38BDF8; font-weight:700;">Logistic Regression:</span><br>
        • Accuracy: <b style="color:#F8FAFC;">{lr_acc*100:.1f}%</b><br>
        • ROC-AUC: <b style="color:#F8FAFC;">{lr_auc:.3f}</b><br><br>
        <span style="color:#38BDF8; font-weight:700;">Decision Tree:</span><br>
        • Accuracy: <b style="color:#F8FAFC;">{dt_acc*100:.1f}%</b><br>
        • ROC-AUC: <b style="color:#F8FAFC;">{dt_auc:.3f}</b><br><br>
        <span style="color:#38BDF8; font-weight:700;">KNN:</span><br>
        • Accuracy: <b style="color:#F8FAFC;">{knn_acc*100:.1f}%</b><br>
        • ROC-AUC: <b style="color:#F8FAFC;">{knn_auc:.3f}</b><br><br>
        <span style="color:#38BDF8; font-weight:700;">Bagging:</span><br>
        • Accuracy: <b style="color:#F8FAFC;">{bagging_acc*100:.1f}%</b><br>
        • ROC-AUC: <b style="color:#F8FAFC;">{bagging_auc:.3f}</b><br><br>
        <span style="color:#38BDF8; font-weight:700;">Boosting:</span><br>
        • Accuracy: <b style="color:#F8FAFC;">{boosting_acc*100:.1f}%</b><br>
        • ROC-AUC: <b style="color:#F8FAFC;">{boosting_auc:.3f}</b>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN HEADER (HOME ONLY)
# -----------------------------------------------------------------------------
if selected_page == nav_items[0]:
    st.markdown("""
    <div class="hero-banner-dark">
        <span class="badge-pill-dark">Enterprise AI Credit Risk Intelligence</span>
        <h1 class="hero-title">FinSecure AI — Loan Default Intelligence</h1>
        <p class="hero-subtitle">
            Real-time risk scoring, default probability estimation, and multi-model credit decision engine powered by
            <b>Logistic Regression</b>, <b>Decision Tree</b>, <b>KNN</b>, <b>Bagging</b>, and <b>Boosting</b> classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 1: INTERACTIVE PREDICTOR (ALL INPUT FIELDS, NO SCROLLBARS / RANGE SLIDERS)
# -----------------------------------------------------------------------------
if selected_page == nav_items[0]:
    st.markdown("### 📝 **Applicant Information & Loan Details**")
    st.caption("Enter applicant attributes directly into the input fields below to compute loan default risk in real-time.")

    # 3 Form Columns
    col_input1, col_input2, col_input3 = st.columns([1, 1, 1])

    with col_input1:
        input_card1 = st.container(border=True)
        input_card1.markdown("<div class=\"card-header-dark\">👤 Personal & Demographics</div>", unsafe_allow_html=True)
        
        # Age as direct Number Input Field
        age = input_card1.number_input(
            "Age (Years)", 
            min_value=18, 
            max_value=100, 
            value=int(current_vals["Age"]) if current_vals else 35, 
            step=1,
            help="Applicant's current age in completed years."
        )
        
        education_options = metadata["categorical_options"]["Education"]
        edu_default_idx = education_options.index(current_vals["Education"]) if current_vals and current_vals["Education"] in education_options else 0
        education = input_card1.selectbox("Education Level", education_options, index=edu_default_idx)

        emp_options = metadata["categorical_options"]["EmploymentType"]
        emp_default_idx = emp_options.index(current_vals["EmploymentType"]) if current_vals and current_vals["EmploymentType"] in emp_options else 0
        employment_type = input_card1.selectbox("Employment Type", emp_options, index=emp_default_idx)

        marital_options = metadata["categorical_options"]["MaritalStatus"]
        mar_default_idx = marital_options.index(current_vals["MaritalStatus"]) if current_vals and current_vals["MaritalStatus"] in marital_options else 0
        marital_status = input_card1.selectbox("Marital Status", marital_options, index=mar_default_idx)

        has_dependents = input_card1.selectbox(
            "Has Dependents?", 
            ["Yes", "No"], 
            index=0 if (current_vals and current_vals["HasDependents"] == "Yes") else 1
        )


    with col_input2:
        input_card2 = st.container(border=True)
        input_card2.markdown("<div class=\"card-header-dark\">💵 Financial Profile</div>", unsafe_allow_html=True)

        # Annual Income as direct Number Input Field
        income = input_card2.number_input(
            "Annual Income ($ USD)", 
            min_value=5000, 
            max_value=1000000, 
            value=int(current_vals["Income"]) if current_vals else 65000, 
            step=1000,
            help="Total verified gross annual income in USD."
        )

        # Credit Score as direct Number Input Field
        credit_score = input_card2.number_input(
            "Credit Score (FICO 300 - 850)", 
            min_value=300, 
            max_value=850, 
            value=int(current_vals["CreditScore"]) if current_vals else 640, 
            step=5,
            help="Applicant FICO credit score."
        )

        # Months Employed as direct Number Input Field
        months_employed = input_card2.number_input(
            "Months Employed", 
            min_value=0, 
            max_value=360, 
            value=int(current_vals["MonthsEmployed"]) if current_vals else 40, 
            step=1,
            help="Number of continuous months in current employment."
        )

        # Number of Active Credit Lines as direct Number Input Field
        num_credit_lines = input_card2.number_input(
            "Number of Active Credit Lines", 
            min_value=1, 
            max_value=10, 
            value=int(current_vals["NumCreditLines"]) if current_vals else 2,
            step=1
        )

        # DTI Ratio as direct Number Input Field
        dti_ratio = input_card2.number_input(
            "Debt-to-Income (DTI) Ratio (e.g., 0.35)", 
            min_value=0.01, 
            max_value=1.00, 
            value=float(current_vals["DTIRatio"]) if current_vals else 0.45, 
            step=0.01,
            format="%.2f",
            help="Monthly debt obligations divided by gross monthly income."
        )


    with col_input3:
        input_card3 = st.container(border=True)
        input_card3.markdown("<div class=\"card-header-dark\">📋 Loan Specifics</div>", unsafe_allow_html=True)

        # Loan Amount as direct Number Input Field
        loan_amount = input_card3.number_input(
            "Requested Loan Amount ($ USD)", 
            min_value=1000, 
            max_value=1000000, 
            value=int(current_vals["LoanAmount"]) if current_vals else 80000, 
            step=2500,
            help="Principal amount requested by the applicant."
        )

        # Loan Term as direct Number Input Field / Selector
        loan_term = input_card3.number_input(
            "Loan Term (Months: 12, 24, 36, 48, 60)", 
            min_value=6, 
            max_value=120, 
            value=int(current_vals["LoanTerm"]) if current_vals else 36,
            step=12
        )

        # Interest Rate as direct Number Input Field
        interest_rate = input_card3.number_input(
            "Annual Interest Rate (% APR)", 
            min_value=1.0, 
            max_value=40.0, 
            value=float(current_vals["InterestRate"]) if current_vals else 12.5, 
            step=0.25,
            format="%.2f",
            help="Fixed annual interest rate percentage."
        )

        purpose_options = metadata["categorical_options"]["LoanPurpose"]
        purp_default_idx = purpose_options.index(current_vals["LoanPurpose"]) if current_vals and current_vals["LoanPurpose"] in purpose_options else 0
        loan_purpose = input_card3.selectbox("Loan Purpose", purpose_options, index=purp_default_idx)

        has_mortgage = input_card3.selectbox(
            "Has Existing Mortgage?", 
            ["Yes", "No"], 
            index=0 if (current_vals and current_vals["HasMortgage"] == "Yes") else 1
        )

        has_cosigner = input_card3.selectbox(
            "Has Co-Signer / Guarantor?", 
            ["Yes", "No"], 
            index=0 if (current_vals and current_vals["HasCoSigner"] == "Yes") else 1
        )


    # Compile applicant input payload
    applicant_payload = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }

    # Financial Estimates
    monthly_emi = calculate_emi(loan_amount, interest_rate, loan_term)
    total_payment = monthly_emi * loan_term
    total_interest = total_payment - loan_amount
    loan_to_income = (loan_amount / income) if income > 0 else 0

    st.markdown("---")

    # Assessment Button / Trigger
    assess_col1, assess_col2, assess_col3 = st.columns([1, 2, 1])
    with assess_col2:
        assess_clicked = st.button("🚀 Run Comprehensive Default Risk Assessment", use_container_width=True)

    # Perform Prediction
    if assess_clicked or True:
        lr_prob, lr_pred = predict_single_applicant(lr_model, preprocessor, applicant_payload, LR_TH)
        dt_prob, dt_pred = predict_single_applicant(dt_model, preprocessor, applicant_payload, DT_TH)
        knn_prob, knn_pred = predict_single_applicant(knn_model, preprocessor, applicant_payload, KNN_TH)
        bagging_prob, bagging_pred = predict_single_applicant(bagging_model, preprocessor, applicant_payload, BAGGING_TH)
        boosting_prob, boosting_pred = predict_single_applicant(boosting_model, preprocessor, applicant_payload, BOOSTING_TH)

        if selected_model_name == "Logistic Regression":
            active_prob = lr_prob
            active_pred = lr_pred
            model_badge = "Logistic Regression Model"
        elif selected_model_name == "Decision Tree":
            active_prob = dt_prob
            active_pred = dt_pred
            model_badge = "Decision Tree Classifier"
        elif selected_model_name == "KNN":
            active_prob = knn_prob
            active_pred = knn_pred
            model_badge = "KNN Classifier"
        elif selected_model_name == "Bagging":
            active_prob = bagging_prob
            active_pred = bagging_pred
            model_badge = "Bagging Classifier"
        elif selected_model_name == "Boosting":
            active_prob = boosting_prob
            active_pred = boosting_pred
            model_badge = "Boosting Classifier"
        else:
            active_prob = (lr_prob + dt_prob + knn_prob + bagging_prob + boosting_prob) / 5
            active_pred = 1 if active_prob >= 0.50 else 0
            model_badge = "Ensemble Consensus (All Models)"

        risk_score = round(active_prob * 100, 1)

        st.markdown("### 📊 **Risk Assessment Results**")

        res_col1, res_col2 = st.columns([1.1, 1])

        with res_col1:
            # Plotly Dark Gauge Chart
            st.markdown(f"**Active Model:** `{model_badge}`")
            fig_gauge = render_risk_gauge_dark(risk_score)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if active_pred == 1 or risk_score >= 50.0:
                st.markdown(f"""
                <div class="verdict-box-danger-dark">
                    <div style="font-size: 2.2rem;">⚠️</div>
                    <div class="verdict-title-dark">HIGH DEFAULT RISK</div>
                    <div class="verdict-desc-dark">
                        Applicant exhibits elevated risk attributes (Estimated Default Risk: <b>{risk_score}%</b>). 
                        It is recommended to decline, demand a creditworthy co-signer, or request additional collateral.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box-safe-dark">
                    <div style="font-size: 2.2rem;">✅</div>
                    <div class="verdict-title-dark">LOW DEFAULT RISK — APPROVED</div>
                    <div class="verdict-desc-dark">
                        Applicant demonstrates strong creditworthiness and safe risk metrics (Estimated Default Risk: <b>{risk_score}%</b>).
                        Approved for standard underwriting.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Metrics Breakdown Row
        st.markdown("<br>", unsafe_allow_html=True)
        met_c1, met_c2, met_c3, met_c4 = st.columns(4)

        with met_c1:
            st.metric("Estimated Monthly EMI", f"${monthly_emi:,.2f}", delta=f"{loan_term} mos term")
        with met_c2:
            st.metric("Total Repayment", f"${total_payment:,.2f}", delta=f"${total_interest:,.0f} Interest")
        with met_c3:
            st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2f}x", delta="Safe < 2.0x" if loan_to_income < 2 else "Elevated", delta_color="normal" if loan_to_income < 2 else "inverse")
        with met_c4:
            st.metric("Model Confidence", f"{max(active_prob, 1 - active_prob)*100:.1f}%", delta=selected_model_name)

        # Multi-Model Comparison Preview
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔬 **Model-by-Model Verdict Comparison**")
        cmp_col1, cmp_col2 = st.columns(2)

        with cmp_col1:
            model_rows = [
                ("Logistic Regression", lr_prob, lr_pred),
                ("Decision Tree", dt_prob, dt_pred),
                ("KNN", knn_prob, knn_pred),
            ]
            for model_name, model_prob, model_pred in model_rows:
                color = "#EF4444" if model_pred == 1 else "#10B981"
                st.markdown(f"""
                <div style="background:#111827; border:1px solid #1F2937; border-radius:12px; padding:15px; margin-bottom:10px; border-left:5px solid {color};">
                    <span style="color:#94A3B8; font-weight:600;">{model_name}:</span><br>
                    • Predicted Risk: <b style="color:#F8FAFC;">{model_prob*100:.1f}%</b><br>
                    • Verdict: <b style="color:{color};">{'DEFAULT RISK (Class 1)' if model_pred == 1 else 'NON-DEFAULT (Class 0)'}</b>
                </div>
                """, unsafe_allow_html=True)

        with cmp_col2:
            model_rows = [
                ("Bagging", bagging_prob, bagging_pred),
                ("Boosting", boosting_prob, boosting_pred),
                ("Ensemble Consensus", active_prob, active_pred),
            ]
            for model_name, model_prob, model_pred in model_rows:
                color = "#EF4444" if model_pred == 1 else "#10B981"
                st.markdown(f"""
                <div style="background:#111827; border:1px solid #1F2937; border-radius:12px; padding:15px; margin-bottom:10px; border-left:5px solid {color};">
                    <span style="color:#94A3B8; font-weight:600;">{model_name}:</span><br>
                    • Predicted Risk: <b style="color:#F8FAFC;">{model_prob*100:.1f}%</b><br>
                    • Verdict: <b style="color:{color};">{'DEFAULT RISK (Class 1)' if model_pred == 1 else 'NON-DEFAULT (Class 0)'}</b>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & ACCURACY DASHBOARD (DARK)
# -----------------------------------------------------------------------------
elif selected_page == nav_items[1]:
    st.markdown("### 📊 **Model Benchmarks & Performance Metrics**")
    st.caption("Comprehensive evaluation on 51,070 unseen test records with class-balanced weighting.")

    metrics = metadata["metrics"]
    lr_m = metrics["logistic_regression"]
    dt_m = metrics["decision_tree"]
    knn_m = metrics["knn"]
    bagging_m = metrics["bagging"]
    boosting_m = metrics["boosting"]

    performance_models = {
        "Logistic Regression": lr_m,
        "Decision Tree": dt_m,
        "KNN": knn_m,
        "Bagging": bagging_m,
        "Boosting": boosting_m,
    }

    # Metric Compass (8 Metrics)
    st.markdown("#### 🧭 **Metric Compass**")
    st.caption("A comprehensive view of the 8 evaluation signals used to benchmark default detection quality across all models.")
    metric_cards = []
    for metric_key, metric_label, metric_format in [
        ("accuracy", "Accuracy", "{:.1%}"),
        ("balanced_accuracy", "Balanced Accuracy", "{:.1%}"),
        ("precision", "Precision", "{:.3f}"),
        ("recall", "Recall", "{:.3f}"),
        ("f1_score", "F1-Score", "{:.3f}"),
        ("roc_auc", "ROC-AUC", "{:.3f}"),
        ("pr_auc", "PR-AUC", "{:.3f}"),
        ("log_loss", "Log Loss", "{:.4f}"),
    ]:
        if metric_key == "log_loss":
            best_model = min(performance_models, key=lambda name: performance_models[name].get(metric_key, 999.0))
        else:
            best_model = max(performance_models, key=lambda name: performance_models[name].get(metric_key, 0.0))
        best_value = performance_models[best_model].get(metric_key, 0.0)
        metric_cards.append(
            f'''<div class="dataset-stat"><div class="dataset-stat-label">{metric_label}</div>
            <div class="dataset-stat-value">{metric_format.format(best_value)}</div>
            <div class="kpi-sub-dark">Best: {best_model}</div></div>'''
        )
    st.markdown('<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-bottom:1.25rem;">' + ''.join(metric_cards) + '</div>', unsafe_allow_html=True)

    # Model Ranking Section
    st.markdown("#### 🏆 **Overall Model Ranking**")
    st.caption(
        "Composite ranking score weights: Balanced Accuracy (20%), Recall (20%), PR-AUC (20%), "
        "ROC-AUC (15%), F1-Score (15%), Precision (5%), and Accuracy (5%)."
    )

    ranking_rows = []
    for model_name, model_metrics in performance_models.items():
        pr_val = model_metrics.get("pr_auc", 0.0)
        loss_val = model_metrics.get("log_loss", 0.0)
        train_acc = model_metrics.get("train_metrics", {}).get("train_accuracy", model_metrics["accuracy"])
        fit_status = model_metrics.get("fit_status", "Good Fit")
        cv_info = model_metrics.get("cv_5fold", {})
        cv_mean = cv_info.get("cv_roc_auc_mean", model_metrics["roc_auc"])
        cv_std = cv_info.get("cv_roc_auc_std", 0.0)

        ranking_score = (
            model_metrics["balanced_accuracy"] * 0.20
            + model_metrics["recall"] * 0.20
            + pr_val * 0.20
            + model_metrics["roc_auc"] * 0.15
            + model_metrics["f1_score"] * 0.15
            + model_metrics["precision"] * 0.05
            + model_metrics["accuracy"] * 0.05
        )
        ranking_rows.append({
            "Model": model_name,
            "Ranking Score": ranking_score,
            "Accuracy": model_metrics["accuracy"],
            "Train Accuracy": train_acc,
            "Fit Status": fit_status,
            "5-Fold CV ROC-AUC": f"{cv_mean:.3f} ± {cv_std:.3f}",
            "Balanced Accuracy": model_metrics["balanced_accuracy"],
            "Precision": model_metrics["precision"],
            "Recall": model_metrics["recall"],
            "F1-Score": model_metrics["f1_score"],
            "ROC-AUC": model_metrics["roc_auc"],
            "PR-AUC": pr_val,
            "Log Loss": loss_val,
        })

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        by="Ranking Score", ascending=False
    ).reset_index(drop=True)

    st.markdown('''<div class="ranking-head"><div>Rank</div><div>Model</div><div>Score</div><div>Accuracy</div><div>Balanced</div><div>Precision</div><div>Recall</div><div>F1</div><div>ROC-AUC</div><div>PR-AUC</div></div>''', unsafe_allow_html=True)
    ranking_cards = []
    for rank, row in enumerate(ranking_df.to_dict("records"), start=1):
        best_tag = '<span style="background:rgba(56,189,248,0.2);color:#38BDF8;padding:2px 7px;border-radius:6px;font-size:0.68rem;font-weight:800;margin-left:6px;border:1px solid rgba(56,189,248,0.4);">★ #1 BEST MODEL</span>' if rank == 1 else ''
        ranking_cards.append(f'''<div class="ranking-card"><div class="ranking-metrics">
            <div class="rank-badge">{rank}</div>
            <div><div class="ranking-model">{row["Model"]}{best_tag}</div><div class="ranking-score">Composite score {row["Ranking Score"]:.4f}</div></div>
            <div class="ranking-value">{row["Ranking Score"]:.4f}</div>
            <div class="ranking-value"><span>Accuracy</span>{row["Accuracy"]:.1%}</div>
            <div class="ranking-value"><span>Balanced</span>{row["Balanced Accuracy"]:.1%}</div>
            <div class="ranking-value"><span>Precision</span>{row["Precision"]:.3f}</div>
            <div class="ranking-value"><span>Recall</span>{row["Recall"]:.3f}</div>
            <div class="ranking-value"><span>F1</span>{row["F1-Score"]:.3f}</div>
            <div class="ranking-value"><span>ROC-AUC</span>{row["ROC-AUC"]:.3f}</div>
            <div class="ranking-value"><span>PR-AUC</span>{row["PR-AUC"]:.3f}</div>
        </div></div>''')
    st.markdown(''.join(ranking_cards), unsafe_allow_html=True)

    # Task 5 Detailed Comparison Table
    with st.expander("📋 **View Complete Evaluation & Cross-Validation Table**", expanded=True):
        st.dataframe(ranking_df, use_container_width=True)

    st.markdown("---")

    # Interactive Curves: ROC and PR Curves
    st.markdown("#### 📈 **Receiver Operating Characteristic (ROC) & PR Curves**")
    st.caption("Visualizing classification power, true positive trade-offs, and precision-recall across decision thresholds.")
    
    curve_col1, curve_col2 = st.columns(2)

    color_palette = {
        "Logistic Regression": "#38BDF8",
        "Decision Tree": "#F59E0B",
        "KNN": "#A855F7",
        "Bagging": "#10B981",
        "Boosting": "#EC4899",
    }

    with curve_col1:
        fig_roc = go.Figure()
        # Add random guess baseline
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='#64748B', dash='dash', width=1.5),
            name='Random Chance (AUC=0.50)'
        ))
        for m_name, m_data in performance_models.items():
            roc_pts = m_data.get("roc_curve", {})
            if roc_pts and "fpr" in roc_pts and "tpr" in roc_pts:
                fig_roc.add_trace(go.Scatter(
                    x=roc_pts["fpr"],
                    y=roc_pts["tpr"],
                    mode='lines',
                    line=dict(color=color_palette.get(m_name, "#38BDF8"), width=2.5),
                    name=f'{m_name} (AUC={m_data["roc_auc"]:.3f})'
                ))
        fig_roc.update_layout(
            title="<b>ROC Curves Comparison (AUC)</b>",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.45, y=0.08, bgcolor='rgba(15,23,42,0.85)', bordercolor='#334155', borderwidth=1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with curve_col2:
        fig_pr = go.Figure()
        for m_name, m_data in performance_models.items():
            pr_pts = m_data.get("pr_curve", {})
            if pr_pts and "recall" in pr_pts and "precision" in pr_pts:
                fig_pr.add_trace(go.Scatter(
                    x=pr_pts["recall"],
                    y=pr_pts["precision"],
                    mode='lines',
                    line=dict(color=color_palette.get(m_name, "#38BDF8"), width=2.5),
                    name=f'{m_name} (PR-AUC={m_data.get("pr_auc", 0):.3f})'
                ))
        fig_pr.update_layout(
            title="<b>Precision-Recall (PR) Curves</b>",
            xaxis_title="Recall (Default Class)",
            yaxis_title="Precision (Default Class)",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.45, y=0.85, bgcolor='rgba(15,23,42,0.85)', bordercolor='#334155', borderwidth=1),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # Overfitting / Underfitting & Cross-Validation Diagnostic Section
    st.markdown("#### 🔬 **Diagnostics: Overfitting Checks & 5-Fold CV Stability**")
    diag_c1, diag_c2, diag_c3 = st.columns(3)
    
    with diag_c1:
        st.markdown("""
        <div class="dataset-stat">
            <div class="dataset-stat-label">Overfitting Check Rule</div>
            <div style="font-size:0.92rem;color:#E2E8F0;margin-top:0.35rem;line-height:1.5;">
                • <b>Train >> Test</b>: Overfitting<br>
                • <b>Both Low</b>: Underfitting<br>
                • <b>Train ≈ Test</b>: Good Fit (Generalizable)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with diag_c2:
        st.markdown("""
        <div class="dataset-stat">
            <div class="dataset-stat-label">5-Fold CV Objective</div>
            <div style="font-size:0.92rem;color:#E2E8F0;margin-top:0.35rem;line-height:1.5;">
                • Measures out-of-fold generalization across 5 random splits.<br>
                • Low standard deviation (spread < 0.01) indicates robust stability.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with diag_c3:
        st.markdown("""
        <div class="dataset-stat">
            <div class="dataset-stat-label">Log Loss / Entropy</div>
            <div style="font-size:0.92rem;color:#E2E8F0;margin-top:0.35rem;line-height:1.5;">
                • Quantifies probability calibration error.<br>
                • Lower log loss indicates sharper, well-calibrated confidence.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Confusion Matrix Explorer
    st.markdown("#### 🔍 **Confusion Matrices on Test Set (51,070 Samples)**")
    
    cm_selected_model = st.selectbox(
        "Select Model to Inspect Confusion Matrix:",
        list(performance_models.keys()),
        index=0,
    )
    
    cm_data = np.array(performance_models[cm_selected_model]["confusion_matrix"])
    tn, fp, fn, tp = cm_data[0][0], cm_data[0][1], cm_data[1][0], cm_data[1][1]
    
    cm_c1, cm_c2 = st.columns([1.2, 1])
    
    with cm_c1:
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['Non-Default (0)', 'Default (1)'],
            y=['Non-Default (0)', 'Default (1)'],
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"{cm_selected_model} Confusion Matrix",
            template="plotly_dark"
        )
        fig_cm.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with cm_c2:
        st.markdown(f"""
        <div class="glass-card-dark" style="padding:1.2rem;margin-top:0.8rem;">
            <div class="card-header-dark">📊 {cm_selected_model} Diagnostic Breakdown</div>
            <div style="font-size:0.88rem;color:#CBD5E1;line-height:1.7;">
                • <b>True Negatives (TN):</b> {tn:,} ({tn/(tn+fp)*100:.1f}% non-default accuracy)<br>
                • <b>False Positives (FP):</b> {fp:,} (Safe borrowers flagged)<br>
                • <b>False Negatives (FN):</b> {fn:,} (Missed defaults - critical risk)<br>
                • <b>True Positives (TP):</b> {tp:,} (Defaults caught: {tp/(tp+fn)*100:.1f}% recall)<br>
                • <b>Log Loss:</b> <span style="color:#38BDF8;font-weight:700;">{performance_models[cm_selected_model].get('log_loss', 0):.4f}</span><br>
                • <b>Fit Verdict:</b> <span style="color:#10B981;font-weight:700;">{performance_models[cm_selected_model].get('fit_status', 'Good Fit')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature Importance / Top Risk Drivers (Dark)
    st.markdown("#### 📈 **Key Feature Influences (Drivers of Default)**")
    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        lr_coefs = metadata["feature_analysis"]["logistic_regression_coefficients"]
        df_lr_coefs = pd.DataFrame(list(lr_coefs.items()), columns=["Feature", "Coefficient"]).sort_values(by="Coefficient", ascending=True)
        df_lr_top = pd.concat([df_lr_coefs.head(6), df_lr_coefs.tail(6)]).drop_duplicates()
        
        fig_coef = px.bar(
            df_lr_top,
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Coefficient",
            color_continuous_scale="RdBu_r",
            title="Logistic Regression Feature Coefficients",
            template="plotly_dark"
        )
        fig_coef.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_coef, use_container_width=True)

    with feat_col2:
        dt_imp = metadata["feature_analysis"]["decision_tree_importances"]
        df_dt_imp = pd.DataFrame(list(dt_imp.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False).head(10)

        fig_imp = px.bar(
            df_dt_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
            title="Decision Tree Feature Importances (Top 10)",
            template="plotly_dark"
        )
        fig_imp.update_layout(height=380, yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: BATCH PREDICTION & CSV ANALYZER (DARK)
# -----------------------------------------------------------------------------
elif selected_page == nav_items[2]:
    st.markdown("### 📁 **Batch Loan Application Assessment**")
    st.caption("Upload a CSV file containing multiple loan applicant records to generate instant default risk scores and export reports.")

    # Option to load test samples
    sample_df = load_sample_test_data()
    use_sample = st.checkbox("🧪 **Load pre-packaged sample dataset (200 test applicants)**", value=True)

    uploaded_file = st.file_uploader("Or upload your custom applicant CSV file:", type=["csv"])

    df_to_score = None
    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"✓ Uploaded CSV with {len(df_to_score):,} records.")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    elif use_sample and sample_df is not None:
        df_to_score = sample_df.copy()

    if df_to_score is not None:
        st.markdown(f"**Preview of Dataset to Score ({len(df_to_score)} rows):**")
        st.dataframe(df_to_score.head(6), use_container_width=True)

        if st.button("⚡ Score All Applicants in Batch"):
            with st.spinner("Scoring records with Machine Learning models..."):
                clean_df = df_to_score.copy()
                if "Actual_Default" in clean_df.columns:
                    clean_df = clean_df.drop(columns=["Actual_Default"])

                if "LoanID" in clean_df.columns:
                    clean_df = clean_df.drop(columns=["LoanID"])

                processed_batch = preprocessor.transform(clean_df)

                lr_batch_probs = lr_model.predict_proba(processed_batch)[:, 1]
                lr_batch_preds = (lr_batch_probs >= LR_TH).astype(int)

                dt_batch_probs = dt_model.predict_proba(processed_batch)[:, 1]
                dt_batch_preds = (dt_batch_probs >= DT_TH).astype(int)

                knn_batch_probs = knn_model.predict_proba(processed_batch)[:, 1]
                knn_batch_preds = (knn_batch_probs >= KNN_TH).astype(int)

                bagging_batch_probs = bagging_model.predict_proba(processed_batch)[:, 1]
                bagging_batch_preds = (bagging_batch_probs >= BAGGING_TH).astype(int)

                boosting_batch_probs = boosting_model.predict_proba(processed_batch)[:, 1]
                boosting_batch_preds = (boosting_batch_probs >= BOOSTING_TH).astype(int)

                scored_df = df_to_score.copy()
                scored_df["LR_Risk_Score (%)"] = np.round(lr_batch_probs * 100, 1)
                scored_df["LR_Verdict"] = ["DEFAULT RISK" if p == 1 else "APPROVED" for p in lr_batch_preds]
                scored_df["DT_Risk_Score (%)"] = np.round(dt_batch_probs * 100, 1)
                scored_df["DT_Verdict"] = ["DEFAULT RISK" if p == 1 else "APPROVED" for p in dt_batch_preds]
                scored_df["KNN_Risk_Score (%)"] = np.round(knn_batch_probs * 100, 1)
                scored_df["KNN_Verdict"] = ["DEFAULT RISK" if p == 1 else "APPROVED" for p in knn_batch_preds]
                scored_df["Bagging_Risk_Score (%)"] = np.round(bagging_batch_probs * 100, 1)
                scored_df["Bagging_Verdict"] = ["DEFAULT RISK" if p == 1 else "APPROVED" for p in bagging_batch_preds]
                scored_df["Boosting_Risk_Score (%)"] = np.round(boosting_batch_probs * 100, 1)
                scored_df["Boosting_Verdict"] = ["DEFAULT RISK" if p == 1 else "APPROVED" for p in boosting_batch_preds]

                # Summary Statistics
                st.markdown("---")
                st.markdown("#### 📊 **Batch Summary Metrics**")
                b_c1, b_c2, b_c3, b_c4 = st.columns(4)

                total_apps = len(scored_df)
                lr_defaults = int((lr_batch_preds == 1).sum())
                dt_defaults = int((dt_batch_preds == 1).sum())
                knn_defaults = int((knn_batch_preds == 1).sum())
                bagging_defaults = int((bagging_batch_preds == 1).sum())
                boosting_defaults = int((boosting_batch_preds == 1).sum())
                avg_risk = float(scored_df["LR_Risk_Score (%)"].mean())

                with b_c1:
                    st.metric("Total Applicants", f"{total_apps:,}")
                with b_c2:
                    st.metric("LR Flagged", f"{lr_defaults:,}", f"{lr_defaults/total_apps*100:.1f}%")
                with b_c3:
                    st.metric("DT Flagged", f"{dt_defaults:,}", f"{dt_defaults/total_apps*100:.1f}%")
                with b_c4:
                    st.metric("Avg LR Risk", f"{avg_risk:.1f}%")

                b_c5, b_c6, b_c7 = st.columns(3)
                with b_c5:
                    st.metric("KNN Flagged", f"{knn_defaults:,}", f"{knn_defaults/total_apps*100:.1f}%")
                with b_c6:
                    st.metric("Bagging Flagged", f"{bagging_defaults:,}", f"{bagging_defaults/total_apps*100:.1f}%")
                with b_c7:
                    st.metric("Boosting Flagged", f"{boosting_defaults:,}", f"{boosting_defaults/total_apps*100:.1f}%")

                # Distribution Chart
                st.markdown("<br>", unsafe_allow_html=True)
                fig_dist = px.histogram(
                    scored_df, 
                    x="LR_Risk_Score (%)", 
                    nbins=25, 
                    color="LR_Verdict",
                    color_discrete_map={"APPROVED": "#10B981", "DEFAULT RISK": "#EF4444"},
                    title="Distribution of Predicted Risk Scores (Logistic Regression)",
                    template="plotly_dark"
                )
                fig_dist.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_dist, use_container_width=True)

                # Display Scored Dataframe
                st.markdown("#### 📋 **Scored Applications Table**")
                st.dataframe(scored_df, use_container_width=True)

                # Export CSV
                csv_data = scored_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Loan Applications (CSV)",
                    data=csv_data,
                    file_name="scored_loan_applications.csv",
                    mime="text/csv"
                )

# -----------------------------------------------------------------------------
# TAB 4: DATASET & EDA INSIGHTS (DARK)
# -----------------------------------------------------------------------------
elif selected_page == nav_items[3]:
    st.markdown("### 📈 **Exploratory Data Analysis (EDA) Insights**")
    st.caption("Key statistical discoveries and relationships from the 255,347 loan applications dataset.")

    st.markdown("#### 🗂️ **Dataset Overview**")
    dataset_stats = [
        ("Records", f"{metadata['total_rows']:,}"),
        ("Features", f"{len(metadata['numerical_columns']) + len(metadata['categorical_columns'])}"),
        ("Test Set", f"{metadata['metrics']['logistic_regression']['test_samples']:,}"),
        ("Default Rate", f"{metadata['target_distribution']['default_rate']:.1%}"),
    ]
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-bottom:1.5rem;">'
        + ''.join(
            f'<div class="dataset-stat"><div class="dataset-stat-label">{label}</div><div class="dataset-stat-value">{value}</div></div>'
            for label, value in dataset_stats
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    eda_c1, eda_c2 = st.columns(2)

    with eda_c1:
        st.markdown("""
        <div class="glass-card-dark">
            <div class="card-header-dark">📊 Loan Purpose Distribution</div>
        """, unsafe_allow_html=True)
        
        purp_df = pd.DataFrame({
            "Loan Purpose": ["Business", "Home", "Education", "Other", "Auto"],
            "Records": [51298, 51286, 51005, 50914, 50844]
        })
        fig_purp = px.pie(purp_df, values="Records", names="Loan Purpose", color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4, template="plotly_dark")
        fig_purp.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_purp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with eda_c2:
        st.markdown("""
        <div class="glass-card-dark">
            <div class="card-header-dark">🎓 Education vs Default Proportions</div>
        """, unsafe_allow_html=True)
        
        edu_df = pd.DataFrame({
            "Education": ["High School", "Bachelor's", "Master's", "PhD"],
            "Default Rate (%)": [13.2, 11.8, 10.9, 10.4]
        })
        fig_edu = px.bar(edu_df, x="Education", y="Default Rate (%)", color="Default Rate (%)", color_continuous_scale="Blues", template="plotly_dark")
        fig_edu.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_edu, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    eda_c3, eda_c4 = st.columns(2)

    with eda_c3:
        st.markdown("""
        <div class="glass-card-dark">
            <div class="card-header-dark">💳 Credit Score vs Default Rate Trend</div>
        """, unsafe_allow_html=True)
        
        cs_ranges = ["300-450", "451-550", "551-650", "651-750", "751-850"]
        cs_rates = [22.4, 16.1, 11.2, 7.8, 4.2]
        df_cs = pd.DataFrame({"FICO Range": cs_ranges, "Default Rate (%)": cs_rates})
        fig_cs = px.line(df_cs, x="FICO Range", y="Default Rate (%)", markers=True, title="Default Rate drops sharply as Credit Score increases", template="plotly_dark")
        fig_cs.update_traces(line_color="#38BDF8", line_width=3)
        fig_cs.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cs, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with eda_c4:
        st.markdown("""
        <div class="glass-card-dark">
            <div class="card-header-dark">📉 Debt-to-Income (DTI) Impact</div>
        """, unsafe_allow_html=True)
        
        dti_bins = ["0.1-0.3 (Low)", "0.3-0.5 (Moderate)", "0.5-0.7 (Elevated)", "0.7-0.9 (High)"]
        dti_rates = [6.8, 9.7, 13.9, 18.5]
        df_dti = pd.DataFrame({"DTI Bracket": dti_bins, "Default Rate (%)": dti_rates})
        fig_dti = px.bar(df_dti, x="DTI Bracket", y="Default Rate (%)", color="Default Rate (%)", color_continuous_scale="Reds", template="plotly_dark")
        fig_dti.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dti, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 5: ARCHITECTURE & DOCUMENTATION (DARK)
# -----------------------------------------------------------------------------
elif selected_page == nav_items[4]:
    st.markdown("### ℹ️ **System Architecture & Machine Learning Pipeline**")
    
    st.markdown("""
    #### 🏗️ **End-to-End Pipeline Architecture **
    1. **Data Ingestion & Cleaning**:
       - Dataset containing **255,347 loan applications** and 18 attributes.
       - Target default class rate: **11.6%** (severe class imbalance).
       - Zero data leakage: Identifiers (`LoanID`) excluded prior to modeling.
    2. **Stratified Splitting & Leakage Prevention**:
       - 64% Training Split (163,422 rows), 16% Validation Split (40,855 rows), 20% Untouched Test Split (51,070 rows).
       - Preprocessing pipelines fitted strictly on training data.
    3. **Imbalance Handling & Cross-Validation**:
       - 5-Fold Stratified Cross-Validation for stability and out-of-fold generalization.
       - SMOTE minority oversampling and balanced class weighting applied within training folds.
    4. **Hyperparameter Tuning & Decision Threshold Optimization**:
       - RandomizedSearchCV across regularization (`C`), tree depths, min leaf samples, leaf nodes, and learning rates.
       - Per-model probability decision threshold tuning on validation set maximizing balanced default detection.
    5. **Evaluation Suite**:
       - Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Log Loss, Confusion Matrix, and ROC curves.
    """)

    st.markdown("---")

    st.markdown("""
    #### 🚀 **What Else Can We Do for Increasing Model Accuracy?**
    *Comprehensive roadmap for enterprise-grade performance gains:*

    1. **Non-Linear Feature Interactions & Financial Ratios**:
       - *Payment-to-Income (PTI)*: Monthly EMI divided by monthly income.
       - *Residual Discretionary Income*: Monthly income minus estimated living expenses and loan installment.
       - *Non-Linear Credit Score Bucketing*: Polynomial features combining `CreditScore * DTIRatio` and `LoanAmount / (MonthsEmployed + 1)`.
       - *Target Encoding / Weight of Evidence (WoE)*: Encoding high-cardinality combinations (e.g., `EmploymentType + LoanPurpose`).

    2. **Advanced Ensemble Stacking & Blending**:
       - Train a **Meta-Classifier (StackingClassifier)** that takes out-of-fold predicted probabilities from Logistic Regression, Random Forest, HistGradientBoosting, and KNN as inputs to a calibrated Logistic Regression meta-learner.
       - Stacking combines linear, tree-based, and neighborhood boundaries to capture complementary decision surfaces.

    3. **Deep Tabular Neural Architectures**:
       - **FT-Transformer (Feature Tokenizer Transformer)**: Self-attention over categorical and numerical embeddings.
       - **TabNet / NODE**: Sparse attention selection specifically designed for tabular datasets of 250K+ records.

    4. **Cost-Sensitive Learning & Focal Loss**:
       - In credit risk, a False Negative (missed default of $80,000) costs significantly more than a False Positive (declining a safe borrower who pays interest).
       - Implement **Custom Asymmetric Loss Functions** or **Focal Loss** to penalize high-risk defaults heavily during gradient descent.

    5. **Probability Calibration (Isotonic Regression & Platt Scaling)**:
       - Calibrate raw boosting and tree probabilities using `CalibratedClassifierCV(method='isotonic')` to drive Log Loss down and make default risk percentages exact probabilities.

    6. **Bayesian Optimization with Optuna**:
       - Upgrade grid/random searches to **Tree-structured Parzen Estimator (TPE)** over 100+ trials with multivariate pruning.
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px;">
        FinSecure AI © 2026 | Enterprise Credit Risk Platform | Streamlit & Scikit-Learn Engine
    </div>
    """, unsafe_allow_html=True)
