from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import mahalanobis 

app = Flask(__name__)

# Load the trained model and encoders
model = load_model('models/crop_yield_ann.keras')
scaler = joblib.load('models/scaler.pkl')
crop_encoder = joblib.load('models/crop_encoder.pkl')
region_encoder = joblib.load('models/region_encoder.pkl')
soil_encoder = joblib.load('models/soil_encoder.pkl')
weather_encoder = joblib.load('models/weather_encoder.pkl')

# the features used in the model
FEATURES = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition', 
            'Temperature_Celsius', 'Rainfall_mm', 'Fertilizer_Used', 
            'Irrigation_Used', 'Days_to_Harvest']

# PART 1: Drift Analysis Setup
# Load training data for drift analysis
training_data = pd.read_csv('data/crop_yield.csv')
training_features = training_data[['Temperature_Celsius', 'Rainfall_mm', 'Days_to_Harvest']]

# Calculate mean and covariance of training data
train_mean = training_features.mean().values
train_cov = np.cov(training_features.T)
train_cov_inv = np.linalg.pinv(train_cov)

def check_drift(input_data):
    """
    Check if new data is different from training data
    input_data: array of 9 features (as the model expects)
    """
    # Take only numerical features (temperature, rainfall, days)
    numerical_input = np.array([input_data[4], input_data[5], input_data[8]])
    
    # Calculate Mahalanobis distance
    mahalanobis_dist = mahalanobis(numerical_input, train_mean, train_cov_inv)
    
    # Threshold (can be adjusted)
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


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load data from the interface
        data = request.get_json()
        
        # Convert numerical data
        temperature = float(data['temperature'])
        rainfall = float(data['rainfall'])
        fertilizer = int(data['fertilizer'])
        irrigation = int(data['irrigation'])
        days = int(data['days'])
        
        # Convert categorical data using the encoders
        region = region_encoder.transform([data['region']])[0]
        soil = soil_encoder.transform([data['soil']])[0]
        crop = crop_encoder.transform([data['crop']])[0]
        weather = weather_encoder.transform([data['weather']])[0]
        
        # Prepare the input array
        input_data = np.array([[
            region, soil, crop, weather,
            temperature, rainfall, fertilizer, irrigation, days
        ]])
        
        # PART 2: Drift Analysis Check
        # Check for data drift
        drift_result = check_drift(input_data[0])
        
        # Apply StandardScaler
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled, verbose=0)
        yield_pred = float(prediction[0][0])
        
        # PART 3: Add drift analysis result to the response
        return jsonify({
            'success': True,
            'prediction': round(yield_pred, 2),
            'message': f'✅ Expected Yield: {round(yield_pred, 2)} tons/ha',
            'drift_analysis': drift_result  
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)