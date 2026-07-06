# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import joblib
# from scipy.spatial.distance import mahalanobis 
# import os

# app = Flask(__name__)

# # Load the trained model and encoders
# base_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(base_dir, 'models', 'crop_yield_model.pkl')
# model = joblib.load(model_path)

# scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
# crop_encoder = joblib.load(os.path.join(base_dir, 'models', 'crop_encoder.pkl'))
# region_encoder = joblib.load(os.path.join(base_dir, 'models', 'region_encoder.pkl'))
# soil_encoder = joblib.load(os.path.join(base_dir, 'models', 'soil_encoder.pkl'))
# weather_encoder = joblib.load(os.path.join(base_dir, 'models', 'weather_encoder.pkl'))

# # the features used in the model
# FEATURES = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition', 
#             'Temperature_Celsius', 'Rainfall_mm', 'Fertilizer_Used', 
#             'Irrigation_Used', 'Days_to_Harvest']

# # PART 1: Drift Analysis Setup
# # Load training data for drift analysis
# training_data = pd.read_csv('data/crop_yield.csv')
# training_features = training_data[['Temperature_Celsius', 'Rainfall_mm', 'Days_to_Harvest']]

# # Calculate mean and covariance of training data
# train_mean = training_features.mean().values
# train_cov = np.cov(training_features.T)
# train_cov_inv = np.linalg.pinv(train_cov)

# def check_drift(input_data):
#     """
#     Check if new data is different from training data
#     input_data: array of 9 features (as the model expects)
#     """
#     # Take only numerical features (temperature, rainfall, days)
#     numerical_input = np.array([input_data[4], input_data[5], input_data[8]])
    
#     # Calculate Mahalanobis distance
#     mahalanobis_dist = mahalanobis(numerical_input, train_mean, train_cov_inv)
    
#     # Threshold (can be adjusted)
#     threshold = 10.0
    
#     if mahalanobis_dist > threshold:
#         return {
#             'drift_detected': True,
#             'distance': round(mahalanobis_dist, 2),
#             'message': f'⚠️ Data drift detected! Distance: {mahalanobis_dist:.2f}'
#         }
#     else:
#         return {
#             'drift_detected': False,
#             'distance': round(mahalanobis_dist, 2),
#             'message': f'✅ No drift detected. Distance: {mahalanobis_dist:.2f}'
#         }


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Load data from the interface
#         data = request.get_json()
        
#         # Convert numerical data
#         temperature = float(data['temperature'])
#         rainfall = float(data['rainfall'])
#         fertilizer = int(data['fertilizer'])
#         irrigation = int(data['irrigation'])
#         days = int(data['days'])
        
#         # Convert categorical data using the encoders
#         region = region_encoder.transform([data['region']])[0]
#         soil = soil_encoder.transform([data['soil']])[0]
#         crop = crop_encoder.transform([data['crop']])[0]
#         weather = weather_encoder.transform([data['weather']])[0]
        
#         # Prepare the input array
#         input_data = np.array([[
#             region, soil, crop, weather,
#             temperature, rainfall, fertilizer, irrigation, days
#         ]])
        
#         # PART 2: Drift Analysis Check
#         # Check for data drift
#         drift_result = check_drift(input_data[0])
        
#         # Apply StandardScaler
#         input_scaled = scaler.transform(input_data)
        
#         # Make prediction
#         prediction = model.predict(input_scaled, verbose=0)
#         # yield_pred = float(prediction[0][0])
#         yield_pred = float(prediction[0][0]) if hasattr(prediction, 'ndim') and prediction.ndim > 1 else float(prediction[0])
        
#         # PART 3: Add drift analysis result to the response
#         return jsonify({
#             'success': True,
#             'prediction': round(yield_pred, 2),
#             'message': f'✅ Expected Crop Yield: {round(yield_pred, 2)} tons/ha',
#             'drift_analysis': drift_result  
#         })
        
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# if __name__ == '__main__':
#     app.run()


from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import mahalanobis
from pathlib import Path
import os
import tensorflow as tf
import keras

# ============================================
# Initialize FastAPI App
# ============================================
app = FastAPI(
    title="SafeGuard Ag - Crop Yield Prediction API",
    description="AI-powered crop yield prediction system with drift detection",
    version="1.0.0"
)

# ============================================
# CORS Middleware
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Setup Paths
# ============================================
base_dir = Path(__file__).parent

# ============================================
# Mount Static Files
# ============================================
static_dir = base_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    print("⚠️ Static directory not found!")

# ============================================
# Load Models & Encoders
# ============================================
print("🔄 Loading models...")
print(f"🔍 TensorFlow version: {tf.__version__}")
print(f"🔍 Keras version: {keras.__version__}")

try:
    model = joblib.load(base_dir / "models" / "crop_yield_model.pkl")
    print(f"✅ Model loaded successfully! Type: {type(model)}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise e

try:
    scaler = joblib.load(base_dir / "models" / "scaler.pkl")
    print("✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load scaler: {e}")
    raise e

try:
    crop_encoder = joblib.load(base_dir / "models" / "crop_encoder.pkl")
    region_encoder = joblib.load(base_dir / "models" / "region_encoder.pkl")
    soil_encoder = joblib.load(base_dir / "models" / "soil_encoder.pkl")
    weather_encoder = joblib.load(base_dir / "models" / "weather_encoder.pkl")
    print("✅ All encoders loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load encoders: {e}")
    raise e

print("✅ All models loaded successfully!")

# ============================================
# Drift Analysis Setup
# ============================================
csv_path = base_dir / "data" / "crop_yield.csv"
if csv_path.exists():
    training_data = pd.read_csv(csv_path)
    training_features = training_data[['Temperature_Celsius', 'Rainfall_mm', 'Days_to_Harvest']]
    train_mean = training_features.mean().values
    train_cov = np.cov(training_features.T)
    train_cov_inv = np.linalg.pinv(train_cov)
    print("✅ Drift analysis data loaded!")
else:
    print(f"⚠️ CSV file not found: {csv_path}")
    train_mean = np.array([25, 100, 120])
    train_cov_inv = np.eye(3)

# ============================================
# Pydantic Models
# ============================================
class PredictionRequest(BaseModel):
    region: str
    soil: str
    crop: str
    weather: str
    temperature: float
    rainfall: float
    fertilizer: int
    irrigation: int
    days: int

# ============================================
# Helper Functions
# ============================================
def check_drift(input_data):
    """Check if new data is different from training data"""
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

def safe_transform(encoder, value):
    """Safe transform with error handling"""
    try:
        return encoder.transform([value])[0]
    except Exception as e:
        raise ValueError(f"Value '{value}' not found in encoder")

# ============================================
# API Endpoints
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    try:
        with open(base_dir / "templates" / "index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading index.html: {e}</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "drift_monitoring": "active"
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict crop yield based on input parameters"""
    try:
        # Convert categorical data using encoders
        region = safe_transform(region_encoder, request.region)
        soil = safe_transform(soil_encoder, request.soil)
        crop = safe_transform(crop_encoder, request.crop)
        weather = safe_transform(weather_encoder, request.weather)
        
        # Prepare input array
        input_data = np.array([[
            region, soil, crop, weather,
            request.temperature, request.rainfall,
            request.fertilizer, request.irrigation, request.days
        ]])
        
        # Check for data drift
        drift_result = check_drift(input_data[0])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)
        yield_pred = float(prediction[0][0])
        
        return {
            'success': True,
            'prediction': round(yield_pred, 2),
            'message': f'✅ Expected Crop Yield: {round(yield_pred, 2)} tons/ha',
            'drift_analysis': drift_result
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================
# Run the App (Commented for Vercel)
# ============================================
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 5000))
#     uvicorn.run(app, host="0.0.0.0", port=port)