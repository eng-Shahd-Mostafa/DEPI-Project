import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="SafeGuard Ag", page_icon="🌾", layout="centered")

# تنسيق CSS للتصميم النظيف (Google-style)
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    h1 { color: #202124; font-family: 'Segoe UI', sans-serif; text-align: center; font-weight: 700; }
    .stButton>button { 
        background-color: #1A73E8; 
        color: white; 
        border-radius: 20px; 
        border: none; 
        padding: 10px 24px; 
        font-weight: 600;
    }
    .stTextInput, .stNumberInput, .stSelectbox { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# العنوان
st.title("🌾 SafeGuard Ag: Crop Yield Prediction")
st.markdown("---")

# تحميل الموديلات
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    soil_enc = joblib.load('models/soil_encoder.pkl')
    weather_enc = joblib.load('models/weather_encoder.pkl')
    region_enc = joblib.load('models/region_encoder.pkl')
    crop_enc = joblib.load('models/crop_encoder.pkl')
    model = tf.keras.models.load_model('models/crop_yield_ann.keras')
    return scaler, soil_enc, weather_enc, region_enc, crop_enc, model

scaler, soil_enc, weather_enc, region_enc, crop_enc, model = load_models()

# تقسيم المدخلات
col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("Region", region_enc.classes_)
    soil = st.selectbox("Soil Type", soil_enc.classes_)
    crop = st.selectbox("Crop", crop_enc.classes_)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

with col2:
    temp = st.number_input("Temperature (°C)", min_value=0.0)
    weather = st.selectbox("Weather Condition", weather_enc.classes_)
    fertilizer = st.selectbox("Fertilizer Used", [True, False])
    irrigation = st.selectbox("Irrigation Used", [True, False])

# حقل الـ Days to Harvest يأخذ سطر كامل
harvest_days = st.number_input("Days to Harvest", min_value=0)

st.markdown("<br>", unsafe_allow_html=True) # مسافة قبل الزر

# زر التوقع في المنتصف باستخدام تقنية الأعمدة
c1, c2, c3 = st.columns([1, 1, 1])

with c2:
    predict_button = st.button("Predict Yield", use_container_width=True)

# منطق التوقع
if predict_button:
    input_data = pd.DataFrame({
        'Region': region_enc.transform([region]),
        'Soil_Type': soil_enc.transform([soil]),
        'Crop': crop_enc.transform([crop]),
        'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temp],
        'Fertilizer_Used': [int(fertilizer)],
        'Irrigation_Used': [int(irrigation)],
        'Weather_Condition': weather_enc.transform([weather]),
        'Days_to_Harvest': [harvest_days]
    })
    
    prediction = model.predict(input_data)
    
    st.success(f"### Predicted Crop Yield: {prediction[0][0]:.2f} tons/hectare")