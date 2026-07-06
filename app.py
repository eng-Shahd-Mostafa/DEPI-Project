import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from drift_analysis import DataDriftAnalyzer

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .stApp {
            background: linear-gradient(180deg, #f5f7fa 0%, #e8ecf1 100%);
        }
        
        .main-container {
            max-width: 2000px;
            margin: 0 auto;
            padding: 24px 32px 40px;
        }
        
        /* ===== Header ===== */
        .header-section {
            background: linear-gradient(135deg, #1a73e8 0%, #6c5ce7 40%, #a855f7 100%);
            border-radius: 28px;
            padding: 48px 56px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 12px 48px rgba(26, 115, 232, 0.25);
        }
        
        .header-section::before {
            content: '🌾';
            position: absolute;
            top: -20%;
            right: -5%;
            font-size: 300px;
            opacity: 0.06;
            transform: rotate(-15deg);
        }
        
        .header-content {
            position: relative;
            z-index: 2;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 24px;
        }
        
        .header-icon-wrapper {
            width: 80px;
            height: 80px;
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            border: 2px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .header-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: white;
            margin: 0;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .header-subtitle {
            color: rgba(255,255,255,0.85);
            font-size: 1.05rem;
            margin: 4px 0 0 0;
            font-weight: 400;
        }
        
        .header-badge-group {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .header-badge {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(20px);
            padding: 10px 24px;
            border-radius: 50px;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .header-badge.gold {
            background: rgba(255,215,0,0.15);
            border-color: rgba(255,215,0,0.3);
        }
        
        /* ===== Stats Cards ===== */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 32px;
        }
        
        .stat-card {
            background: white;
            padding: 22px 25px;
            border-radius: 20px;
            border: none;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            border-radius: 20px 20px 0 0;
        }
        
        .stat-card:nth-child(1)::before { background: linear-gradient(90deg, #1a73e8, #6c5ce7); }
        .stat-card:nth-child(2)::before { background: linear-gradient(90deg, #34a853, #66bb6a); }
        .stat-card:nth-child(3)::before { background: linear-gradient(90deg, #fbbc04, #ffa726); }
        .stat-card:nth-child(4)::before { background: linear-gradient(90deg, #ea4335, #ef5350); }
        
        .stat-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.08);
        }
        
        .stat-number {
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #1a2332 0%, #4a5a6e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stat-label {
            color: #5f6b7a;
            font-size: 0.85rem;
            margin: 6px 0 0 0;
            font-weight: 500;
        }
        
        .stat-icon {
            font-size: 28px;
            display: block;
            margin-bottom: 4px;
        }
        
        /* ===== Tabs ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: #f0f2f5;
            border-radius: 16px;
            padding: 6px;
            border: none;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 10px 28px;
            font-weight: 600;
            color: #5f6b7a;
            font-size: 1.4rem;
            transition: all 0.3s ease;
            border: none;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.5);
            color: #1a73e8;
        }
        
        .stTabs [aria-selected="true"] {
            background: white !important;
            color: #1a73e8 !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            font-weight: 600;
        }
        
        /* ===== Form Elements ===== */
        .stSelectbox label,
        .stNumberInput label {
            font-weight: 600 !important;
            color: #1a2332 !important;
            font-size: 1.4rem !important;
            margin-bottom: 8px !important;
        }
        
        .stSelectbox [data-baseweb="select"], 
        .stNumberInput [data-baseweb="input"] {
            border-radius: 14px !important;
            border: 1.5px solid #e8ecf0 !important;
            transition: all 0.3s ease !important;
            background: #fafbfc !important;
        }
        
        .stSelectbox [data-baseweb="select"]:hover,
        .stNumberInput [data-baseweb="input"]:hover {
            border-color: #1a73e8 !important;
            background: white !important;
        }
        
        .stSelectbox [data-baseweb="select"]:focus,
        .stNumberInput [data-baseweb="input"]:focus {
            border-color: #1a73e8 !important;
            box-shadow: 0 0 0 4px rgba(26, 115, 232, 0.08) !important;
            background: white !important;
        }
        
        /* ===== Predict Button ===== */
        .stButton button {
            width: 100% !important;
            padding: 18px 36px !important;
            background: linear-gradient(135deg, #1a73e8 0%, #6c5ce7 60%, #a855f7 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 18px !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 6px 24px rgba(26, 115, 232, 0.3) !important;
            letter-spacing: 0.3px !important;
            position: relative;
            overflow: hidden;
        }
        
        .stButton button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.6s ease;
        }
        
        .stButton button:hover::before {
            left: 100%;
        }
        
        .stButton button:hover {
            transform: translateY(-4px) !important;
            box-shadow: 0 12px 48px rgba(26, 115, 232, 0.4) !important;
        }
        
        .stButton button:active {
            transform: translateY(0px) !important;
        }
        
        /* ===== Result Card ===== */
        .result-card {
            background: white;
            border-radius: 28px;
            padding: 40px 44px;
            box-shadow: 0 12px 48px rgba(52, 168, 83, 0.12);
            animation: slideUp 0.7s cubic-bezier(0.4, 0, 0.2, 1);
            margin-top: 24px;
            border: 2px solid #e8f5e9;
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
            background: linear-gradient(90deg, #34a853, #66bb6a, #a8d5ba);
            border-radius: 28px 28px 0 0;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(40px) scale(0.96);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        .result-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 40px;
            align-items: center;
        }
        
        .result-left {
            display: flex;
            flex-direction: column;
        }
        
        .result-label {
            color: #5f6b7a;
            font-size: 0.9rem;
            font-weight: 500;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .result-value {
            font-size: 5rem;
            font-weight: 800;
            margin: 8px 0 12px 0;
            background: linear-gradient(135deg, #1a73e8 0%, #6c5ce7 50%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -2px;
            line-height: 1;
        }
        
        .result-unit {
            font-size: 1.5rem;
            font-weight: 500;
            color: #5f6b7a;
            -webkit-text-fill-color: #5f6b7a;
        }
        
        .result-status-wrapper {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 8px;
        }
        
        .result-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        
        .result-status.success {
            background: #e8f5e9;
            color: #1e7e34;
            border: 1px solid #a5d6a7;
        }
        
        .result-status.info {
            background: #e3f2fd;
            color: #0d47a1;
            border: 1px solid #90caf9;
        }
        
        .result-status.drift {
            background: #fff3e0;
            color: #e65100;
            border: 1px solid #ffcc80;
        }
        
        .result-right {
            background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%);
            border-radius: 20px;
            padding: 32px 28px;
            text-align: center;
            border: 1px solid #e8f0fe;
        }
        
        .accuracy-number {
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, #34a853 0%, #66bb6a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .accuracy-label {
            color: #5f6b7a;
            font-size: 0.9rem;
            margin: 4px 0 0 0;
            font-weight: 500;
        }
        
        .accuracy-detail {
            margin-top: 14px;
            padding-top: 14px;
            border-top: 1px solid #e8ecf0;
            color: #5f6b7a;
            font-size: 0.85rem;
        }
        
        .accuracy-detail span {
            font-weight: 600;
            color: #1a2332;
        }
        
        /* ===== Analytics - Clean Version ===== */
        .analytics-card {
            background: white;
            border-radius: 24px;
            padding: 28px 32px;
            margin-top: 24px;
            border: 1px solid #f0f0f0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02);
        }
        
        .analytics-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1a2332;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        /* ===== Footer ===== */
        .footer {
            text-align: center;
            padding: 32px 0 8px 0;
            color: #8a94a6;
            border-top: 1px solid #e8ecf0;
            margin-top: 32px;
        }
        
        .footer-title {
            font-weight: 600;
            color: #1a2332;
            font-size: 0.95rem;
            margin: 0;
        }
        
        .footer-sub {
            font-size: 0.8rem;
            color: #aab3c2;
            margin: 4px 0 0 0;
        }
        
        .footer-icons {
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-top: 12px;
        }
        
        .footer-icons span {
            font-size: 20px;
            opacity: 0.5;
            transition: opacity 0.3s ease;
        }
        
        .footer-icons span:hover {
            opacity: 1;
        }
        
        /* ===== Responsive ===== */
        @media (max-width: 992px) {
            .result-grid {
                grid-template-columns: 1fr;
                gap: 24px;
            }
            
            .result-value {
                font-size: 3.5rem;
            }
        }
        
        @media (max-width: 768px) {
            .main-container {
                padding: 16px;
            }
            
            .header-section {
                padding: 28px 24px;
                border-radius: 20px;
            }
            
            .header-content {
                flex-direction: column;
                text-align: center;
            }
            
            .header-left {
                flex-direction: column;
                text-align: center;
            }
            
            .header-title {
                font-size: 1.8rem;
            }
            
            .header-icon-wrapper {
                width: 64px;
                height: 64px;
                font-size: 32px;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }
            
            .stat-card {
                padding: 16px 18px;
            }
            
            .stat-number {
                font-size: 1.8rem;
            }
            
            .result-card {
                padding: 24px 20px;
                border-radius: 20px;
            }
            
            .result-value {
                font-size: 2.8rem;
            }
            
            .result-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 8px 16px;
                font-size: 0.8rem;
            }
            
            .header-badge-group {
                justify-content: center;
            }
            
            .accuracy-number {
                font-size: 2.8rem;
            }
            
            .analytics-card {
                padding: 20px 16px;
            }
        }
        
        @media (max-width: 480px) {
            .stats-grid {
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }
            
            .stat-card {
                padding: 12px 14px;
            }
            
            .stat-number {
                font-size: 1.4rem;
            }
            
            .result-value {
                font-size: 2.2rem;
            }
        }
        
        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f0f2f5;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #1a73e8, #6c5ce7);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #1557b0, #5a4bd1);
        }
        
        /* ===== Streamlit Overrides ===== */
        div[data-testid="stExpander"] {
            border: none !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02) !important;
            border-radius: 20px !important;
            margin-top: 20px !important;
        }
        
        div[data-testid="stExpander"] details {
            border: 1px solid #f0f0f0 !important;
            border-radius: 20px !important;
            background: white !important;
        }
        
        div[data-testid="stExpander"] summary {
            padding: 16px 24px !important;
            font-weight: 600 !important;
            color: #1a2332 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_models():
    models_path = 'models/'
    try:
        return {
            'model': joblib.load(os.path.join(models_path, 'crop_yield_model.pkl')),
            'scaler': joblib.load(os.path.join(models_path, 'scaler.pkl')),
            'region_encoder': joblib.load(os.path.join(models_path, 'region_encoder.pkl')),
            'soil_encoder': joblib.load(os.path.join(models_path, 'soil_encoder.pkl')),
            'crop_encoder': joblib.load(os.path.join(models_path, 'crop_encoder.pkl')),
            'weather_encoder': joblib.load(os.path.join(models_path, 'weather_encoder.pkl'))
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

@st.cache_data
def load_crop_data():
    try:
        return pd.read_csv('data/crop_yield.csv')
    except:
        return None

@st.cache_resource
def init_drift_analyzer():
    df = load_crop_data()
    return DataDriftAnalyzer(reference_data=df) if df is not None else DataDriftAnalyzer()

# Prediction function
def predict_yield(input_data, models):
    try:
        region_enc = models['region_encoder'].transform([input_data['region']])[0]
        soil_enc = models['soil_encoder'].transform([input_data['soil']])[0]
        crop_enc = models['crop_encoder'].transform([input_data['crop']])[0]
        weather_enc = models['weather_encoder'].transform([input_data['weather']])[0]
        
        features = np.array([
            region_enc, soil_enc, crop_enc, weather_enc,
            input_data['temperature'], input_data['rainfall'],
            input_data['fertilizer'], input_data['irrigation'],
            input_data['days']
        ]).reshape(1, -1)
        
        features_scaled = models['scaler'].transform(features)
        prediction_result = models['model'].predict(features_scaled)
        
        try:
            if hasattr(prediction_result, 'item'):
                prediction_value = float(prediction_result.item())
            elif isinstance(prediction_result, np.ndarray):
                prediction_value = float(prediction_result[0])
            else:
                prediction_value = float(prediction_result)
        except:
            prediction_value = float(np.array(prediction_result).flatten()[0])
        
        return prediction_value, 0.94
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

# Yield chart
def create_yield_gauge(value, max_val=200):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': 100, 'increasing': {'color': "#34a853"}, 'decreasing': {'color': "#ea4335"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 50, 'color': '#1a2332', 'family': 'Inter'}},
        title={'text': "Yield (tons/ha)", 'font': {'size': 16, 'color': '#5f6b7a'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "#8a94a6", 
                    'tickfont': {'size': 12, 'color': '#8a94a6'}},
            'bar': {'color': "#1a73e8", 'thickness': 0.4},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e8ecf0",
            'steps': [
                {'range': [0, 50], 'color': '#fce4ec'},
                {'range': [50, 100], 'color': '#fff3e0'},
                {'range': [100, 150], 'color': '#e8f5e9'},
                {'range': [150, 200], 'color': '#e3f2fd'}
            ],
            'threshold': {
                'line': {'color': "#ea4335", 'width': 4},
                'thickness': 0.6,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1a2332", 'family': "Inter"}
    )
    
    return fig

# Main function
def main():
    load_custom_css()
    
    models = load_models()
    if not models:
        return
    
    drift_analyzer = init_drift_analyzer()
    
    # ===== Header =====
    st.markdown("""
    <div class="main-container">
        <div class="header-section">
            <div class="header-content">
                <div class="header-left">
                    <div class="header-icon-wrapper">🌾</div>
                    <div>
                        <h1 class="header-title">Crop Yield Prediction</h1>
                        <p class="header-subtitle">Intelligent AI forecasting for sustainable agriculture</p>
                    </div>
                </div>
                <div class="header-badge-group">
                    <span class="header-badge">🤖 AI Powered</span>
                    <span class="header-badge gold">⭐ 0.94 R²</span>
                    <span class="header-badge">🚀 DEPI</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== Stats =====
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-icon">🌍</span>
            <p class="stat-number">4</p>
            <p class="stat-label">Regions</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-icon">🌱</span>
            <p class="stat-number">5</p>
            <p class="stat-label">Crop Types</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-icon">🏞️</span>
            <p class="stat-number">6</p>
            <p class="stat-label">Soil Types</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-icon">📊</span>
            <p class="stat-number">94%</p>
            <p class="stat-label">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== Form =====
    st.markdown("""
    <br><br>
    <div class="form-title">
        <span class="form-title-icon">📋</span>
        Input Parameters
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌾 Crop Details", "🌤️ Climate Data", "⚙️ Advanced Settings"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            region = st.selectbox("📍 Region", ['North', 'South', 'East', 'West'])
            crop = st.selectbox("🌱 Crop Type", ['Cotton', 'Rice', 'Barley', 'Wheat', 'Maize'])
        with c2:
            soil = st.selectbox("🏞️ Soil Type", ['Sandy', 'Clay', 'Loam', 'Silt', 'Peaty', 'Chalky'])
            days = st.number_input("📅 Days to Harvest", min_value=30, max_value=365, value=120, step=1)
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            weather = st.selectbox("☁️ Weather Condition", ['Sunny', 'Rainy', 'Cloudy'])
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1, format="%.2f")
        with c2:
            rainfall = st.number_input("💧 Rainfall (mm)", min_value=0, max_value=500, value=100, step=1)
    
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fertilizer = st.selectbox("🧪 Fertilizer Used", ['Yes', 'No'])
        with c2:
            irrigation = st.selectbox("💦 Irrigation Used", ['Yes', 'No'])
    
    # ===== Prediction =====
    if st.button("✨ Predict Crop Yield", use_container_width=True):
        input_data = {
            'region': region, 'soil': soil, 'crop': crop, 'weather': weather,
            'temperature': float(temperature), 'rainfall': float(rainfall),
            'fertilizer': 1 if fertilizer == 'Yes' else 0,
            'irrigation': 1 if irrigation == 'Yes' else 0,
            'days': int(days)
        }
        
        try:
            prediction, confidence = predict_yield(input_data, models)
            final_prediction = float(prediction)
            drift_status = drift_analyzer.check_drift(input_data)
            
            # ===== Result =====
            st.markdown(f"""
            <div class="result-card">
                <div class="result-grid">
                    <div class="result-left">
                        <p class="result-label">🌾 Predicted Crop Yield</p>
                        <p class="result-value">{final_prediction:.2f}<span class="result-unit"> tons/ha</span></p>
                        <div class="result-status-wrapper">
                            <span class="result-status success">✅ Prediction Successful</span>
                            <span class="result-status {'info' if not drift_status.get('drift_detected') else 'drift'}">
                                {drift_status.get('message', 'No drift detected')}
                            </span>
                        </div>
                    </div>
                    <div class="result-right">
                        <p class="accuracy-number">R² = {confidence:.2f}</p>
                        <p class="accuracy-label">🎯 Model Accuracy</p>
                        <div class="accuracy-detail">
                            ⚡ <span>{drift_status.get('details', {}).get('warnings_count', 0)}</span> warnings detected
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ===== Advanced Analytics =====
            st.markdown("""
            <div class="analytics-card">
                <div class="analytics-title">
                    <span>📊</span> Advanced Analytics
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            fig = create_yield_gauge(final_prediction)
            st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ {str(e)}")
    
    # ===== Footer =====
    st.markdown("""
    <div class="footer">
        <p class="footer-title">🌾 Crop Yield Prediction System · DEPI Project</p>
        <p class="footer-sub">Powered by AI &amp; Deep Learning Technologies</p>
        <div class="footer-icons">
            <span>🤖</span>
            <span>🧠</span>
            <span>📊</span>
            <span>🌱</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()