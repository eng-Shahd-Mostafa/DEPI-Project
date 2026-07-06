import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import mahalanobis
from pathlib import Path
import os

# ============================================
# Setup Paths
# ============================================
base_dir = Path(__file__).parent

# ============================================
# Load Models & Encoders
# ============================================
@st.cache_resource
def load_models():
    """Load all models and encoders with caching"""
    try:
        # 1. Try loading model from .h5 first (preferred format)
        model_path_h5 = base_dir / "models" / "crop_yield_model.h5"
        model_path_pkl = base_dir / "models" / "crop_yield_model.pkl"
        
        model = None
        
        # Try .h5 first
        if model_path_h5.exists():
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path_h5, compile=False)
                print("✅ Loaded Keras model from .h5")
            except Exception as e:
                print(f"⚠️ Failed to load .h5: {e}")
        
        # If .h5 failed, try .pkl
        if model is None and model_path_pkl.exists():
            try:
                import tensorflow as tf
                # Try loading with joblib
                model = joblib.load(model_path_pkl)
                print("✅ Loaded Keras model from .pkl")
            except Exception as e:
                print(f"⚠️ Failed to load .pkl: {e}")
        
        # If still no model, raise error
        if model is None:
            raise Exception("No model found in either .h5 or .pkl format")
        
        # 2. Load encoders and scaler
        scaler = joblib.load(base_dir / "models" / "scaler.pkl")
        crop_encoder = joblib.load(base_dir / "models" / "crop_encoder.pkl")
        region_encoder = joblib.load(base_dir / "models" / "region_encoder.pkl")
        soil_encoder = joblib.load(base_dir / "models" / "soil_encoder.pkl")
        weather_encoder = joblib.load(base_dir / "models" / "weather_encoder.pkl")
        
        # 3. Load drift analysis data
        csv_path = base_dir / "data" / "crop_yield.csv"
        if csv_path.exists():
            training_data = pd.read_csv(csv_path)
            training_features = training_data[['Temperature_Celsius', 'Rainfall_mm', 'Days_to_Harvest']]
            train_mean = training_features.mean().values
            train_cov = np.cov(training_features.T)
            train_cov_inv = np.linalg.pinv(train_cov)
        else:
            # Fallback values if CSV not found
            print("⚠️ CSV not found, using fallback values")
            train_mean = np.array([25.0, 100.0, 120.0])
            train_cov_inv = np.eye(3)
        
        return {
            'model': model,
            'scaler': scaler,
            'crop_encoder': crop_encoder,
            'region_encoder': region_encoder,
            'soil_encoder': soil_encoder,
            'weather_encoder': weather_encoder,
            'train_mean': train_mean,
            'train_cov_inv': train_cov_inv
        }
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        return None

# ============================================
# Helper Functions
# ============================================
def safe_transform(encoder, value):
    """Safe transform with error handling"""
    try:
        return encoder.transform([value])[0]
    except:
        return 0

def check_drift(input_data, train_mean, train_cov_inv):
    """Check for data drift"""
    numerical_input = np.array([input_data[4], input_data[5], input_data[8]])
    mahalanobis_dist = mahalanobis(numerical_input, train_mean, train_cov_inv)
    threshold = 10.0
    
    if mahalanobis_dist > threshold:
        return {
            'drift_detected': True,
            'distance': round(mahalanobis_dist, 2),
            'message': f'⚠️ Data drift detected! Distance: {mahalanobis_dist:.2f}'
        }
    else:
        return {
            'drift_detected': False,
            'distance': round(mahalanobis_dist, 2),
            'message': f'✅ No drift detected. Distance: {mahalanobis_dist:.2f}'
        }

# ============================================
# Main App
# ============================================
def main():
    # Page config
    st.set_page_config(
        page_title="SafeGuard Ag - Crop Yield Prediction",
        page_icon="🌾",
        layout="wide"
    )
    
    # Load models
    models = load_models()
    if models is None:
        st.stop()
    
    # Header
    st.title("🌾 SafeGuard Ag - Crop Yield Prediction")
    st.markdown("---")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Input Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)
            row3_col1, row3_col2 = st.columns(2)
            
            with row1_col1:
                region = st.selectbox(
                    "🌍 Region",
                    ["North", "South", "East", "West"]
                )
                soil = st.selectbox(
                    "🏞️ Soil Type",
                    ["Sandy", "Clay", "Loam", "Silt", "Peaty", "Chalky"]
                )
            
            with row1_col2:
                crop = st.selectbox(
                    "🌱 Crop Type",
                    ["Cotton", "Rice", "Barley", "Wheat", "Maize"]
                )
                weather = st.selectbox(
                    "☁️ Weather Condition",
                    ["Sunny", "Rainy", "Cloudy"]
                )
            
            with row2_col1:
                temperature = st.number_input(
                    "🌡️ Temperature (°C)",
                    min_value=0.0,
                    max_value=50.0,
                    value=25.0,
                    step=0.1
                )
                rainfall = st.number_input(
                    "💧 Rainfall (mm)",
                    min_value=0,
                    max_value=500,
                    value=100,
                    step=1
                )
            
            with row2_col2:
                fertilizer = st.selectbox(
                    "🧪 Fertilizer Used",
                    [1, 0],
                    format_func=lambda x: "✅ Yes" if x == 1 else "❌ No"
                )
                irrigation = st.selectbox(
                    "💦 Irrigation Used",
                    [1, 0],
                    format_func=lambda x: "✅ Yes" if x == 1 else "❌ No"
                )
            
            with row3_col1:
                days = st.number_input(
                    "📅 Days to Harvest",
                    min_value=30,
                    max_value=300,
                    value=120,
                    step=1
                )
            
            # Submit button
            submitted = st.form_submit_button(
                "🔮 Predict Crop Yield",
                use_container_width=True,
                type="primary"
            )
    
    with col2:
        st.subheader("📊 Results")
        
        # Placeholder for results
        if 'prediction' not in st.session_state:
            st.info("👈 Enter parameters and click 'Predict Crop Yield'")
        else:
            # Display prediction
            result = st.session_state.prediction
            pred_value = result['prediction']
            
            # Main prediction
            st.metric(
                label="🌾 Expected Crop Yield",
                value=f"{pred_value} tons/ha",
                delta=None
            )
            
            # Drift analysis
            st.markdown("---")
            st.subheader("📈 Drift Analysis")
            drift = result['drift_analysis']
            
            if drift['drift_detected']:
                st.warning(f"{drift['message']}")
            else:
                st.success(f"{drift['message']}")
            
            # Additional info
            st.markdown("---")
            st.caption(f"🎯 Model Accuracy: **R² = 0.94**")
    
    # Process prediction
    if submitted:
        try:
            # Transform categorical data
            region_encoded = safe_transform(models['region_encoder'], region)
            soil_encoded = safe_transform(models['soil_encoder'], soil)
            crop_encoded = safe_transform(models['crop_encoder'], crop)
            weather_encoded = safe_transform(models['weather_encoder'], weather)
            
            # Prepare input
            input_data = np.array([[
                region_encoded, soil_encoded, crop_encoded, weather_encoded,
                temperature, rainfall, fertilizer, irrigation, days
            ]])
            
            # Check drift
            drift_result = check_drift(
                input_data[0],
                models['train_mean'],
                models['train_cov_inv']
            )
            
            # Scale and predict
            input_scaled = models['scaler'].transform(input_data)
            prediction = models['model'].predict(input_scaled, verbose=0)
            yield_pred = float(prediction[0][0])
            
            # Store in session state
            st.session_state.prediction = {
                'prediction': round(yield_pred, 2),
                'drift_analysis': drift_result
            }
            
            # Rerun to update display
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()