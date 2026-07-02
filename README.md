# 🌾 SafeGuard Ag: AI-Powered Crop Yield Prediction System

SafeGuard Ag is an intelligent Machine Learning and Deep Learning system designed to accurately predict crop yield based on environmental and agricultural factors.

The project integrates a complete AI pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, noise detection using XGBoost, feature scaling, and an Artificial Neural Network (ANN) built with TensorFlow/Keras.

The goal is to support precision agriculture by providing reliable crop yield predictions that assist farmers and decision-makers in maximizing productivity and optimizing agricultural resources.

---

## 🏗️ Project Structure

```text
SafeGuard-Ag/
├── 📂 data/
│   └── crop_yield.csv                    # Agricultural dataset (10,000+ rows)
│
├── 📂 graphs/                            # 📊 Generated visualizations
│   ├── plot_01_yield_distribution.png    # Yield distribution analysis
│   ├── plot_02_yield_by_crop.png         # Yield by crop type
│   ├── plot_03_yield_by_region.png       # Yield by region
│   ├── plot_04_correlation_matrix.png    # Feature correlation
│   └── plot_05_feature_importance.png    # Feature importance
│   └── ...
│
├── 📂 models/                            # 🧠 Trained artifacts
│   ├── crop_encoder.pkl                  # Crop label encoder
│   ├── region_encoder.pkl                # Region label encoder
│   ├── soil_encoder.pkl                  # Soil type encoder
│   ├── weather_encoder.pkl               # Weather condition encoder
│   ├── scaler.pkl                        # StandardScaler
│   └── crop_yield_ann.keras              # Trained ANN model
│
├── 📂 notebooks/                         # 📓 Jupyter notebooks
│   └── Crop_Yield_Prediction.ipynb       # Main training pipeline
│
├── 📂 static/                            # 🎨 Web assets
│   ├── style.css                         # Custom styling
│   └── script.js                         # Frontend interactivity
│
├── 📂 templates/                         # 📄 HTML templates
│   └── index.html                        # Main web interface
│
├── 📂 venv/                              # 🐍 Python environment
│
├── .gitignore
├── app.py                                # 🚀 Flask application
├── README.md
└── requirements.txt                      # 📦 Dependencies
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
conda create -n DEPI-Project python=3.10 -y

conda activate DEPI-Project
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

## 👥 **Team & Contributors**

<div align="center">

### 🎯 **Project Lead & AI Engineer**

<table align="center">
<tr>
<td align="center">
<img src="https://avatars.githubusercontent.com/eng-Shahd-Mostafa" width="150px" style="border-radius: 50%;" alt="Shahd Mostafa"/>
<br />
<strong>Shahd Mostafa</strong>
<br />
🎓 AI & ML Engineer
<br />
🔬 Deep Learning Specialist
<br />
<br />
<a href="https://github.com/eng-Shahd-Mostafa">
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>
<a href="https://www.linkedin.com/in/engshahdmostafa/">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>
<a href="mailto:eng.shahd.mostafa@gmail.com">
  <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
</a>
</td>
</tr>
</table>

---

## 🚀 **Deployment & Implementation**

### **🌐 Live Demo**

<div align="center">

| Platform | Status | Link |
|----------|--------|------|
| **Local Server** | 🟢 Active | `http://127.0.0.1:5000` |
| **GitHub Repository** | 🟢 Public | [View Repository](https://github.com/eng-Shahd-Mostafa/DEPI-Project) |
| **Documentation** | 🟢 Complete | [README.md](README.md) |

</div>

---

### **📊 Deployment Architecture**

```mermaid
graph TD
    A[User Interface] --> B[Flask Web App]
    B --> C[Prediction API]
    C --> D[Model Loading]
    D --> E[ANN Model]
    D --> F[Preprocessing Pipeline]
    F --> G[Label Encoders]
    F --> H[StandardScaler]
    E --> I[Prediction Result]
    I --> J[JSON Response]
    J --> A
```
---
<div align="center">
<br />
<h5 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 30px; font-weight: 500; margin: 10px 0;">
  Made with ❤️ Shahd Mostafa
</h5>
<br />
</div>