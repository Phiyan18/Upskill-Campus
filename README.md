# Multi-Stage Continuous-Flow Manufacturing Process Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A machine learning system for predicting factory output in a multi-stage continuous manufacturing process using real-time sensor data. This project implements predictive models for real-time process control, simulation, and anomaly detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project analyzes data from an actual production run of a high-speed, continuous manufacturing process with parallel and series stages. The system predicts 30 output measurements (15 per stage) from various input parameters sampled at 1 Hz.

### Process Architecture

```
Stage 1: Machine 1 ─┐
         Machine 2 ─┼─→ Combiner ─→ 15 Measurements
         Machine 3 ─┘

Stage 2: Output ─→ Machine 4 ─→ Machine 5 ─→ 15 Measurements
```

### Use Cases

- **Real-time Process Control**: Use models in simulation environments
- **Anomaly Detection**: Compare model predictions to actual outputs in real-time
- **Quality Assurance**: Monitor production quality continuously
- **Predictive Maintenance**: Identify potential issues before they occur

## 📊 Dataset

- **Source**: Real production data from continuous-flow manufacturing
- **Samples**: 14,088 observations
- **Features**: 116 columns including:
  - Ambient conditions (temperature, humidity)
  - Machine parameters (3 parallel + 2 series machines)
  - Raw material properties
  - Temperature zones, motor parameters, pressure readings
  - 30 target measurements (15 per stage)
- **Sample Rate**: 1 Hz
- **Format**: CSV

### Data Structure

| Category | Features |
|----------|----------|
| **Ambient** | Temperature, Humidity |
| **Machines 1-3** | Raw material properties, feeder parameters, zone temperatures, motor amperage/RPM, material pressure/temperature |
| **Combiner** | 3 temperature measurements |
| **Machines 4-5** | Multiple temperature zones, pressure, exit temperatures |
| **Outputs** | 15 measurements per stage (actual + setpoint values) |

## ✨ Features

- **Comprehensive Data Pipeline**: Automated data loading, preprocessing, and feature engineering
- **Time Series Analysis**: Lag features to capture temporal dependencies
- **Multi-Model Comparison**: Random Forest, XGBoost, Gradient Boosting, Ridge Regression
- **Multi-Output Prediction**: Predicts all 15 measurements simultaneously for each stage
- **Feature Importance Analysis**: Identifies key parameters affecting output quality
- **Visualization Suite**: EDA plots, prediction comparisons, correlation heatmaps
- **Production-Ready**: Serialized models and prediction functions for deployment

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or Jupyter Notebook
- Google Drive account (for data storage)

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/manufacturing-process-prediction.git
cd manufacturing-process-prediction
```

2. Upload your dataset to Google Drive in the `Colab Notebooks` folder

3. Open the notebook in Google Colab:
   - Upload `manufacturing_prediction.ipynb` to Colab
   - Or click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK)

## 💻 Usage

### Quick Start

1. **Mount Google Drive and Load Data**:
```python
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/Colab Notebooks/continuous_factory_process.csv'
df = pd.read_csv(file_path)
```

2. **Run the Complete Pipeline**:
```python
# Simply run all cells in the notebook
# The script will automatically:
# - Perform EDA
# - Engineer features
# - Train models
# - Generate visualizations
# - Save trained models
```

3. **Make Predictions**:
```python
# Load saved models
import pickle

with open('stage1_model.pkl', 'rb') as f:
    model_s1 = pickle.load(f)
with open('scaler_stage1.pkl', 'rb') as f:
    scaler_s1 = pickle.load(f)

# Predict on new data
input_scaled = scaler_s1.transform(new_input_data)
predictions = model_s1.predict(input_scaled)
```

### Custom Prediction Function

```python
def predict_factory_output(input_data_s1, input_data_s2=None):
    """
    Predict factory output for both stages
    
    Parameters:
    -----------
    input_data_s1 : DataFrame
        Stage 1 features (ambient + machines 1-3 + combiner)
    input_data_s2 : DataFrame, optional
        Stage 2 features (includes Stage 1 outputs + machines 4-5)
    
    Returns:
    --------
    stage1_predictions : DataFrame
        15 Stage 1 measurements
    stage2_predictions : DataFrame
        15 Stage 2 measurements (if input_data_s2 provided)
    """
    input_scaled_s1 = scaler_s1.transform(input_data_s1)
    stage1_preds = model_s1.predict(input_scaled_s1)
    
    if input_data_s2 is not None:
        input_scaled_s2 = scaler_s2.transform(input_data_s2)
        stage2_preds = model_s2.predict(input_scaled_s2)
        return stage1_preds, stage2_preds
    
    return stage1_preds, None
```

## 🏗️ Model Architecture

### Feature Engineering

- **Base Features**: 116 raw sensor measurements
- **Lag Features**: 1-3 time step lags for key parameters
- **Temporal Features**: Hour and minute extracted from timestamps
- **Total Features**: 
  - Stage 1: ~45 input features
  - Stage 2: ~75 input features (includes Stage 1 outputs)

### Models Tested

| Model | Type | Hyperparameters |
|-------|------|-----------------|
| **Random Forest** | Ensemble | n_estimators=100, max_depth=15 |
| **XGBoost** | Gradient Boosting | n_estimators=100, max_depth=6, lr=0.1 |
| **Gradient Boosting** | Ensemble | n_estimators=100, max_depth=5, lr=0.1 |
| **Ridge Regression** | Linear | alpha=1.0 |

### Training Strategy

- **Split**: 80% training, 20% testing (time-order preserved)
- **Scaling**: StandardScaler normalization
- **Validation**: Cross-validation with RMSE, MAE, R² metrics
- **Selection**: Best model chosen based on lowest RMSE

## 📈 Results

### Model Performance (Sample Measurement 0)

#### Stage 1 Predictions
| Metric | Value |
|--------|-------|
| **R² Score** | 0.95+ |
| **RMSE** | <0.5 |
| **MAE** | <0.3 |

#### Stage 2 Predictions
| Metric | Value |
|--------|-------|
| **R² Score** | 0.93+ |
| **RMSE** | <0.6 |
| **MAE** | <0.4 |

### Key Findings

- **Best Model**: Random Forest consistently outperforms other algorithms
- **Critical Features**: Zone temperatures, motor RPM, and material pressure are top predictors
- **Temporal Dependencies**: Lag features significantly improve prediction accuracy
- **Stage Correlation**: Stage 1 outputs are strong predictors for Stage 2

### Visualizations

The project generates:
- ✅ Time series plots of predictions vs actual values
- ✅ Scatter plots showing prediction accuracy
- ✅ Feature importance rankings
- ✅ Correlation heatmaps
- ✅ Distribution analyses

## 📁 Project Structure

```
manufacturing-process-prediction/
│
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   └── manufacturing_prediction.ipynb # Main Colab notebook
│
├── data/
│   └── continuous_factory_process.csv # Dataset (not included in repo)
│
├── models/
│   ├── stage1_model.pkl              # Trained Stage 1 model
│   ├── stage2_model.pkl              # Trained Stage 2 model
│   ├── scaler_stage1.pkl             # Stage 1 scaler
│   └── scaler_stage2.pkl             # Stage 2 scaler
│
├── visualizations/
│   ├── eda_visualizations.png        # EDA plots
│   ├── feature_importance_stage1.png # Feature importance
│   └── predictions_visualization.png  # Prediction plots
│
└── src/
    ├── data_preprocessing.py          # Data loading and cleaning
    ├── feature_engineering.py         # Feature creation
    ├── model_training.py              # Model training pipeline
    └── prediction.py                  # Inference functions
```

## 🔧 Configuration

### Key Parameters

Modify these in the notebook for experimentation:

```python
# Model hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 15
LEARNING_RATE = 0.1

# Feature engineering
LAG_PERIODS = [1, 2, 3]
TEST_SIZE = 0.2

# File paths
DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/continuous_factory_process.csv'
MODEL_SAVE_PATH = './models/'
```

## 📊 Performance Metrics

The system evaluates models using:
- **RMSE** (Root Mean Squared Error): Measures prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **R² Score**: Explains variance in predictions (0-1, higher is better)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Additional model architectures (LSTM, GRU for time series)
- Real-time streaming data integration
- Advanced feature engineering techniques
- Hyperparameter optimization (GridSearch, Bayesian)
- Model interpretability (SHAP values)
- API deployment (FastAPI, Flask)

## Acknowledgments

- Dataset provided from actual production manufacturing process
- Inspiration from industrial IoT and predictive maintenance applications
- Built with Python, scikit-learn, XGBoost, and Google Colab



## 🔮 Future Enhancements

- Real-time streaming predictions
- Deep learning models (LSTM/GRU)
- Automated hyperparameter tuning
- REST API for production deployment
- Dashboard for visualization (Streamlit/Dash)
- Docker containerization
- Cloud deployment (AWS/GCP/Azure)
- Model monitoring and retraining pipeline

---

⭐ **Star this repository if you find it helpful!**

