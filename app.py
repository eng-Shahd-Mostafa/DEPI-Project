import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# استيراد DataDriftAnalyzer من ملف drift_analysis.py
# ============================================
from drift_analysis import DataDriftAnalyzer

# Page configuration
st.set_page_config(
    page_title="SafeGuard Ag - Smart Yield Predictor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Updated with bolder colors
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&display=swap');
    
    /* Background - bolder gradient */
    .stApp {
        background: linear-gradient(160deg, #D4E4FF 0%, #C8E6C8 100%);
        background-attachment: fixed;
    }
    
    /* Main container with stronger border */
    .main {
        background: rgba(255, 255, 255, 0.98) !important;
        border-radius: 32px !important;
        padding: 2.5rem !important;
        box-shadow: 
            0 8px 32px rgba(0, 20, 40, 0.12),
            0 0 0 2px rgba(26, 115, 232, 0.15) !important;
        border: 2px solid #1A73E8 !important;
        margin: 0.5rem 0;
    }
    
    /* Header - bolder gradient */
    h1 {
        background: linear-gradient(135deg, #0D47A1 0%, #1B5E20 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Google Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        text-align: center;
        padding: 0.3rem 0;
        letter-spacing: -0.02em;
        margin-bottom: 0 !important;
    }
    
    .brand-badge {
        text-align: center;
        font-family: 'Google Sans', sans-serif;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #1A73E8;
        opacity: 0.8;
        margin-top: -0.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Divider - bolder */
    hr {
        margin: 1.5rem 0 1.8rem 0;
        border: none;
        height: 3px;
        background: linear-gradient(90deg, 
            transparent 0%,
            #1A73E8 30%,
            #34A853 70%,
            transparent 100%
        );
        border-radius: 10px;
        opacity: 0.5;
    }
    
    /* Section header with bolder colors */
    .section-header {
        font-family: 'Google Sans', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #0D47A1;
        margin-bottom: 1.2rem;
        padding: 0.7rem 0 0.7rem 1.2rem;
        border-left: 6px solid #0D47A1;
        background: rgba(13, 71, 161, 0.08);
        border-radius: 0 12px 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Card containers with stronger borders */
    .css-1r6slb0, .css-1d391kg {
        background: white !important;
        border-radius: 18px !important;
        padding: 1.5rem 1.8rem !important;
        border: 2px solid #BBDEFB !important;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.08) !important;
        transition: all 0.3s ease !important;
        margin-bottom: 0.8rem;
    }
    
    .css-1r6slb0:hover, .css-1d391kg:hover {
        border-color: #1A73E8 !important;
        box-shadow: 0 6px 20px rgba(13, 71, 161, 0.15) !important;
        transform: translateY(-2px);
    }
    
    /* Labels - bolder */
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        font-family: 'Google Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #0D47A1 !important;
        letter-spacing: 0.02em;
        margin-bottom: 0.3rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        opacity: 0.9;
    }
    
    /* Input fields with bolder borders */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput input, 
    .stTextInput input {
        border-radius: 14px !important;
        border: 2px solid #90CAF9 !important;
        transition: all 0.3s ease !important;
        background: #F8FBFF !important;
        font-family: 'Google Sans', sans-serif !important;
        padding: 11px 16px !important;
        font-size: 0.95rem !important;
        color: #1A1A1A !important;
        height: 48px !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:hover,
    .stNumberInput input:hover, 
    .stTextInput input:hover {
        border-color: #1A73E8 !important;
        background: white !important;
        box-shadow: 0 0 0 3px rgba(13, 71, 161, 0.08) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:focus-within,
    .stNumberInput input:focus, 
    .stTextInput input:focus {
        border-color: #0D47A1 !important;
        box-shadow: 0 0 0 4px rgba(13, 71, 161, 0.12) !important;
        background: white !important;
    }
    
    /* Dropdown - bolder */
    div[data-baseweb="select"] ul {
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
        border: 2px solid #BBDEFB !important;
        background: white !important;
        padding: 8px !important;
    }
    
    div[data-baseweb="select"] li {
        padding: 12px 20px !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
        font-family: 'Google Sans', sans-serif !important;
        margin: 2px 0 !important;
        color: #1A1A1A !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="select"] li:hover {
        background: #E3F2FD !important;
        color: #0D47A1 !important;
    }
    
    div[data-baseweb="select"] li[aria-selected="true"] {
        background: linear-gradient(135deg, #0D47A1, #1A73E8) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Number input buttons - bolder */
    .stNumberInput button {
        background: #F0F7FF !important;
        border: 2px solid #90CAF9 !important;
        border-radius: 10px !important;
        color: #0D47A1 !important;
        transition: all 0.2s ease !important;
        font-size: 1.1rem !important;
        padding: 0 14px !important;
        min-width: 40px !important;
        height: 40px !important;
        font-weight: 700 !important;
    }
    
    .stNumberInput button:hover {
        background: #E3F2FD !important;
        border-color: #0D47A1 !important;
        color: #0D47A1 !important;
        box-shadow: 0 2px 8px rgba(13, 71, 161, 0.15) !important;
    }
    
    /* Button - bolder */
    .stButton>button {
        background: linear-gradient(135deg, #0D47A1 0%, #1A73E8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 18px 40px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        font-family: 'Google Sans', sans-serif !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 6px 24px rgba(13, 71, 161, 0.35) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 40px rgba(13, 71, 161, 0.45) !important;
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) scale(0.98) !important;
    }
    
    /* Result Card - bolder */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-container {
        background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E9 100%) !important;
        border-radius: 28px !important;
        border: 3px solid #2E7D32 !important;
        padding: 2rem 2.5rem !important;
        margin-top: 2rem !important;
        box-shadow: 0 8px 32px rgba(27, 94, 32, 0.2) !important;
        animation: slideIn 0.5s ease forwards;
        text-align: center;
    }
    
    .result-value {
        font-family: 'Google Sans', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        margin: 0;
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .result-unit {
        font-family: 'Google Sans', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: #1B5E20;
        -webkit-text-fill-color: #1B5E20;
    }
    
    .result-detail {
        font-family: 'Google Sans', sans-serif;
        color: #1B5E20;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        padding-top: 0.8rem;
        border-top: 2px solid rgba(27, 94, 32, 0.2);
        font-weight: 500;
    }
    
    .result-footer {
        font-family: 'Google Sans', sans-serif;
        color: #33691E;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        font-weight: 500;
        opacity: 0.7;
    }
    
    /* Status badges - bolder */
    .badge-success {
        display: inline-block;
        background: rgba(27, 94, 32, 0.15);
        color: #1B5E20;
        padding: 6px 20px;
        border-radius: 20px;
        font-family: 'Google Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        border: 2px solid rgba(27, 94, 32, 0.25);
    }
    
    .badge-warning {
        display: inline-block;
        background: rgba(255, 152, 0, 0.15);
        color: #E65100;
        padding: 6px 20px;
        border-radius: 20px;
        font-family: 'Google Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        border: 2px solid rgba(255, 152, 0, 0.25);
    }
    
    .badge-error {
        display: inline-block;
        background: rgba(211, 47, 47, 0.15);
        color: #B71C1C;
        padding: 6px 20px;
        border-radius: 20px;
        font-family: 'Google Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        border: 2px solid rgba(211, 47, 47, 0.25);
    }
    
    /* Drift card */
    .drift-container {
        background: white !important;
        border-radius: 18px !important;
        padding: 1.2rem 1.8rem !important;
        border: 2px solid #BBDEFB !important;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.08) !important;
        margin-top: 1rem !important;
        animation: slideIn 0.5s ease forwards;
    }
    
    .drift-title {
        font-family: 'Google Sans', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: #0D47A1;
        margin-bottom: 0.8rem;
    }
    
    .drift-warning-item {
        font-family: 'Google Sans', sans-serif;
        font-size: 0.9rem;
        color: #E65100;
        padding: 0.4rem 0;
        border-bottom: 1px solid #FFF3E0;
    }
    
    .drift-safe-item {
        font-family: 'Google Sans', sans-serif;
        font-size: 0.9rem;
        color: #1B5E20;
        padding: 0.4rem 0;
        border-bottom: 1px solid #E8F5E9;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main {
            padding: 1.5rem !important;
            border-radius: 24px !important;
        }
        h1 {
            font-size: 2.2rem !important;
        }
        .stButton>button {
            padding: 14px 28px !important;
            font-size: 0.95rem !important;
        }
        .result-value {
            font-size: 2.5rem !important;
        }
        .css-1r6slb0, .css-1d391kg {
            padding: 1.2rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Load reference data with fallback to sample data
@st.cache_data
def load_real_reference_data():
    try:
        df = pd.read_csv('data/crop_yield.csv')
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv('crop_yield.csv')
            return df
        except FileNotFoundError:
            st.warning("⚠️ Reference data file 'crop_yield.csv' not found. Using backup sample data for drift analysis.")
            return pd.DataFrame({
                'Temperature': [25, 28, 22, 30, 20, 26, 24, 29, 23, 27],
                'Rainfall': [50, 60, 45, 70, 40, 55, 48, 65, 42, 58],
                'Days_to_Harvest': [120, 115, 130, 110, 125, 118, 122, 112, 128, 116],
                'Region': ['North', 'South', 'East', 'West', 'Central'] * 2,
                'Soil_Type': ['Loamy', 'Sandy', 'Clay', 'Silty', 'Peaty'] * 2,
                'Crop_Type': ['Wheat', 'Rice', 'Corn', 'Soybean', 'Cotton'] * 2,
                'Weather_Condition': ['Sunny', 'Cloudy', 'Rainy', 'Windy', 'Foggy'] * 2
            })

reference_df = load_real_reference_data()

# Initialize Drift Analyzer with the reference data
drift_analyzer = DataDriftAnalyzer(reference_data=reference_df)

# Header
st.markdown("""
    <div style="text-align: center; margin-bottom: -0.8rem;">
        <span style="font-size: 2.8rem;">🌾</span>
    </div>
""", unsafe_allow_html=True)
st.title("SafeGuard Ag")
st.markdown("<div class='brand-badge'>✦ SMART AGRICULTURE PLATFORM ✦</div>", unsafe_allow_html=True)
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    soil_enc = joblib.load('models/soil_encoder.pkl')
    weather_enc = joblib.load('models/weather_encoder.pkl')
    region_enc = joblib.load('models/region_encoder.pkl')
    crop_enc = joblib.load('models/crop_encoder.pkl')
    model = tf.keras.models.load_model('models/crop_yield_ann.h5', compile=False)
    return scaler, soil_enc, weather_enc, region_enc, crop_enc, model

scaler, soil_enc, weather_enc, region_enc, crop_enc, model = load_models()

# Farm Details Section
st.markdown('<div class="section-header">📋 Farm Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    region = st.selectbox("📍 Region", region_enc.classes_)
    soil = st.selectbox("🧪 Soil Type", soil_enc.classes_)
    crop = st.selectbox("🌱 Crop", crop_enc.classes_)

with col2:
    temp = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1, format="%.1f", value=25.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=0.1, format="%.1f", value=50.0)
    weather = st.selectbox("☁️ Weather Condition", weather_enc.classes_)

# Additional Parameters Section
st.markdown('<div class="section-header" style="margin-top: 1.2rem;">📅 Additional Parameters</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    fertilizer = st.selectbox("🧴 Fertilizer Used", ["No", "Yes"])
with col4:
    irrigation = st.selectbox("💧 Irrigation Used", ["No", "Yes"])
with col5:
    harvest_days = st.number_input("📅 Days to Harvest", min_value=1, max_value=365, step=1, value=120, format="%d")

fertilizer_bool = fertilizer == "Yes"
irrigation_bool = irrigation == "Yes"

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    predict_button = st.button("🚀 Predict Crop Yield", use_container_width=True)

st.markdown("---")

# Prediction Logic
if predict_button:
    input_data = pd.DataFrame({
        'Region': region_enc.transform([region]),
        'Soil_Type': soil_enc.transform([soil]),
        'Crop': crop_enc.transform([crop]),
        'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temp],
        'Fertilizer_Used': [int(fertilizer_bool)],
        'Irrigation_Used': [int(irrigation_bool)],
        'Weather_Condition': weather_enc.transform([weather]),
        'Days_to_Harvest': [harvest_days]
    })
    
    # Prepare raw input for drift analysis
    raw_input_for_drift = {
        'temperature': temp,
        'rainfall': rainfall,
        'days': harvest_days
    }
    
    with st.spinner("🌱 Analyzing field data..."):
        # Check for data drift
        drift_result = drift_analyzer.check_drift(raw_input_for_drift)
        
        prediction = model.predict(input_data, verbose=0)
        yield_value = prediction[0][0]
    
    # Result Display with bolder colors
    st.markdown(f"""
        <div class="result-container">
            <div style="margin-bottom: 0.5rem;">
                <span style="font-family: 'Google Sans', sans-serif; font-weight: 600; font-size: 1.1rem; color: #1B5E20;">Prediction Result</span>
            </div>
            <div class="result-value">
                {yield_value:.2f} 
                <span class="result-unit">tons/ha</span>
            </div>
            <div style="margin-top: 0.5rem;">
                <span class="badge-success">✅ Prediction successful!</span>
            </div>
            <div class="result-detail">
                📊 Estimated yield based on provided parameters
            </div>
            <div class="result-footer">
                SafeGuard Ag • {datetime.now().strftime('%B %d, %Y')}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Data Drift Analysis Section
    st.markdown('<div class="section-header" style="margin-top: 1.5rem;">📊 Data Drift Analysis</div>', unsafe_allow_html=True)
    
    # Determine badge class and icon based on drift detection
    if drift_result['drift_detected']:
        badge_class = "badge-warning"
        badge_icon = "⚠️"
        drift_status = "Drift Detected"
    else:
        badge_class = "badge-success"
        badge_icon = "✅"
        drift_status = "No Drift Detected"
    
    # Display drift status
    st.markdown(f"""
        <div class="drift-container">
            <div class="drift-title">
                🔍 Data Drift Analysis
                <span style="float: right;">
                    <span class="{badge_class}">{badge_icon} {drift_status}</span>
                </span>
            </div>
            <div style="margin-top: 0.5rem;">
                <span style="font-family: 'Google Sans', sans-serif; font-size: 0.9rem; color: #555;">
                    <strong>Status:</strong> {drift_result['message']}
                </span>
            </div>
            <div style="margin-top: 0.5rem;">
                <span style="font-family: 'Google Sans', sans-serif; font-size: 0.85rem; color: #777;">
                    <strong>Warnings:</strong> {drift_result['details']['warnings_count']}
                </span>
            </div>
    """, unsafe_allow_html=True)
    
    # Display warnings if any
    if drift_result['warnings']:
        st.markdown('<div style="margin-top: 0.8rem;"><strong style="font-family: Google Sans, sans-serif; color: #E65100;">⚠️ Warnings:</strong></div>', unsafe_allow_html=True)
        for warning in drift_result['warnings']:
            st.markdown(f'<div class="drift-warning-item">• {warning}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="drift-safe-item" style="margin-top: 0.5rem; color: #1B5E20;">
                ✅ All input parameters are within expected ranges
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show input summary in organized format
    with st.expander("📋 View Input Summary", expanded=False):
        summary_data = {
            "Parameter": ["Region", "Soil Type", "Crop", "Temperature", "Rainfall", 
                        "Weather", "Fertilizer", "Irrigation", "Days to Harvest"],
            "Value": [region, soil, crop, f"{temp}°C", f"{rainfall} mm", 
                    weather, fertilizer, irrigation, f"{harvest_days} days"]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium")
            }
        )