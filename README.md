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
├── 📂 graphs/
│   └── plot_01_yield_distribution.png
│   └── plot_02_yield_by_crop.png
│   └── plot_03_yield_by_region.png
│   └── plot_04_fertilizer_irrigation.png
│   └── ...
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
│
├── .gitignore
├── README.md
└── requirements.txt
```
---
## 📌 Key Features

- ✅ **Data Preprocessing** – Handling missing values, outliers, and duplicates.
- ✅ **Exploratory Data Analysis (EDA)** – Statistical analysis, correlation matrices, and rich visualizations.
- ✅ **Feature Engineering** – Encoding categorical variables and scaling numerical features.
- ✅ **Noise Detection** – Using XGBoost to detect and remove noisy data points (top 10% residuals).
- ✅ **Deep Learning Model** – Artificial Neural Network (ANN) with:
  - 3 Dense layers (128, 64, 32 neurons)
  - Batch Normalization & Dropout for regularization
  - Adam optimizer with learning rate scheduling
- ✅ **Model Evaluation** – R² Score, RMSE, MAE, and stability analysis across multiple runs.
- ✅ **Model Persistence** – Save trained models, encoders, and scaler using `joblib` and `keras`.
- ✅ **Flask Web Application** – User-friendly interface for real-time predictions.
- ✅ **Modern UI/UX** – Responsive design with smooth animations and gradient aesthetics.

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
## 🛠️ Tech Stack

### **Data Science & Machine Learning**
| Tool/Library | Version | Purpose |
|--------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **Pandas** | 2.0.3 | Data manipulation & analysis |
| **NumPy** | 1.24.3 | Numerical computations |
| **Scikit-learn** | 1.3.0 | Preprocessing, encoders, metrics |
| **TensorFlow/Keras** | 2.13.0 | Deep learning model (ANN) |
| **XGBoost** | 1.7.6 | Noise detection & feature importance |
| **Matplotlib** | 3.7.2 | Data visualization |
| **Seaborn** | 0.12.2 | Statistical visualizations |
| **Joblib** | 1.3.1 | Model serialization |

### **Web Development & Deployment**
| Tool/Library | Version | Purpose |
|--------------|---------|---------|
| **Flask** | 2.3.2 | Web framework |
| **HTML5/CSS3** | - | Frontend structure & styling |
| **JavaScript** | - | Client-side interactivity |
| **Google Fonts** | - | Typography (Inter & Tajawal) |

---

## 📊 Model Performance

The ANN model achieved the following evaluation metrics after rigorous training and noise cleaning:

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9426 |
| **RMSE** | 0.4001 tons/ha |
| **MAE** | 0.3323 tons/ha |
| **Residual Std** | 0.3999 |

### Stability Analysis (5 Runs)
| Metric | Mean | Std |
|--------|------|-----|
| **R²** | 0.941727 | 0.001041 |
| **RMSE** | 0.402539 | 0.003557 |
| **MAE** | 0.33408 | 0.00240 |

The model demonstrates **high stability** and **consistency** across multiple training runs.

---

## 📈 Data Preprocessing Pipeline

### 1. **Data Cleaning**
- Removed negative yield values
- Detected and handled outliers using IQR method
- Removed top 10% noisy data points using XGBoost residual analysis

### 2. **Feature Engineering**
- Encoded categorical features:
  - `Region`
  - `Soil_Type`
  - `Crop`
  - `Weather_Condition`
- Converted boolean features to integers:
  - `Fertilizer_Used`
  - `Irrigation_Used`
- Standardized numerical features using `StandardScaler`

### 3. **Train/Test Split**
- 80% training, 20% testing
- Random state: 42 for reproducibility

---
## 🧠 Model Architecture

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 128)               1280
batch_normalization (BN)     (None, 128)               512
dropout (Dropout)            (None, 128)               0
dense_1 (Dense)              (None, 64)                8256
batch_normalization_1 (BN)   (None, 64)                256
dropout_1 (Dropout)          (None, 64)                0
dense_2 (Dense)              (None, 32)                2080
batch_normalization_2 (BN)   (None, 32)                128
dense_3 (Dense)              (None, 1)                 33
=================================================================
Total params: 12,545
Trainable params: 12,097
Non-trainable params: 448
_________________________________________________________________
---

## ⚙️ Prerequisites

Before running the project, ensure the following software is installed:

- Python 3.10+
- Jupyter Notebook
- Anaconda (Optional)
- Git

---

## 🚀 Getting Started & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/eng-Shahd-Mostafa/DEPI-Project.git

cd DEPI-Project
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

Using Conda:

```bash
conda create -n safeguard-ag python=3.10 -y

conda activate safeguard-ag
```

Or using venv:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

Open one of the following notebooks:

- `Crop_Yield_Prediction.ipynb`
- `Crop_Yield_Prediction_Algo.ipynb`

## ▶️ Running the Project

After installing the required dependencies, launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/Crop_Yield_Prediction.ipynb
```

Run all notebook cells sequentially to:

- Load the dataset
- Perform data preprocessing
- Analyze the data
- Train the ANN model
- Evaluate model performance
- Generate crop yield predictions