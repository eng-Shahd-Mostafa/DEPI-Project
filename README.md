# 🌾 SafeGuard Ag: AI-Powered Crop Yield Prediction System

SafeGuard Ag is an intelligent Machine Learning and Deep Learning system designed to accurately predict crop yield based on environmental and agricultural factors.

The project integrates a complete AI pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, noise detection using XGBoost, feature scaling, and an Artificial Neural Network (ANN) built with TensorFlow/Keras.

The goal is to support precision agriculture by providing reliable crop yield predictions that assist farmers and decision-makers in maximizing productivity and optimizing agricultural resources.

---

## 🏗️ Project Structure

```text
SafeGuard-Ag/
├── 📂 data/
│   └── crop_yield.csv                 # Agricultural dataset
│
├── 📂 models/
│   ├── crop_encoder.pkl               # Crop label encoder
│   ├── region_encoder.pkl             # Region label encoder
│   ├── soil_encoder.pkl               # Soil type encoder
│   ├── weather_encoder.pkl            # Weather condition encoder
│   ├── scaler.pkl                     # Feature scaler
│   └── crop_yield_ann.keras           # Trained ANN model
│
├── 📂 notebooks/
│   ├── Crop_Yield_Prediction.ipynb
│   └── Crop_Yield_Prediction_Algo.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```
---

## 🎯 Milestones

### Milestone 1: Data Collection, Exploration, and Preprocessing

- Load agricultural dataset.
- Handle missing values.
- Remove duplicates.
- Detect outliers.
- Encode categorical variables.
- Normalize numerical features.

### Milestone 2: Exploratory Data Analysis

- Select and fine-tune a model: FaceNet, VGG-Face, DeepFace, or custom CNN.
- Statistical Analysis.
- Distribution Analysis.
- Correlation Matrix.
- Feature Importance.
- Data Visualization.

### Milestone 3: Model Development & Evaluation

- Deploy the model via Flask API.
- Feature Engineering.
- XGBoost Noise Detection.
- Train/Test Split.
- Feature Scaling.
- ANN Development.
- MAE, RMSE, R² Score, Error Analysis

### Milestone 4: Deployment & Future Improvements

- Model Saving.
- Prediction Pipeline.
- Explainable AI.
- Dashboard Integration.

### Milestone 5: Final Documentation and Presentation

- Full project report covering data, model, deployment, and monitoring.
- Presentation of system architecture and real-world impact.

---

## ⚙️ Prerequisites

Before running the project, ensure the following software is installed:

- Python 3.10+
- Jupyter Notebook
- Anaconda (Optional)
- Git

---

## 🚀 Getting Started & Installation

### 1. Set up the AI Engine (Python)

Open Anaconda Prompt and create the environment for the AI pipeline:

```bash
conda create -n AI python=3.10 -y
conda activate AI
pip install -r requirements.txt
# Or install manually:
# pip install opencv-python deepface tensorflow scipy flask flask-cors requests numpy
```

### 2. Set up the Backend (Node.js)

Open a terminal in the project root directory and install backend dependencies:

```bash
cd web_app/backend
npm install
```

### 3. Set up the Frontend (React.js)

In the same or a new terminal, install the frontend dependencies:

```bash
cd web_app/frontend
npm install
```

---

## 🏃‍♂️ Running the System

To launch the complete system (AI Stream + Backend API + React Dashboard), simply run the batch script from your project root:

```bash
start.bat
```