# Customer Churn Predictor 🔮

An end-to-end Machine Learning web application designed to predict telecom customer churn. Built with **XGBoost**, **FastAPI**, and an interactive web interface (**HTML/CSS/JS**), this platform helps businesses identify high-risk customers and implement proactive retention strategies.

---

## 🚀 Key Features

* **Machine Learning Pipeline**: Trained XGBoost binary classifier with hyperparameter tuning, feature engineering (Tenure Groups, Service Ratios, Total Services), and probability scoring.
* **REST API**: FastAPI backend powering real-time churn prediction, batch predictions, and feature importance analysis endpoints.
* **Interactive Frontend**: Modern web dashboard to input customer details (contract, internet service, payment method, tenure, monthly charges) and receive instant risk scores with actionable insights.
* **Synthetic / Real Data Generator**: Embedded data pipeline (`generate_data.py`) to generate realistic Telco Customer Churn datasets for benchmark testing.

---

## 📊 Model Performance & Metrics

The tuned XGBoost model achieved the following performance metrics on test evaluation:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **72.7%** | Overall percentage of accurate predictions |
| **Precision** | **68.5%** | Correctly identified churning customers out of total predicted churns |
| **Recall** | **68.5%** | Sensitivity in catching actual churning customers |
| **F1 Score** | **68.5%** | Harmonic balance between Precision and Recall |
| **ROC-AUC** | **81.6%** | Area under the Receiver Operating Characteristic curve |

---

## 📂 Project Structure

```tree
Customer churn predictor/
├── data/
│   └── customer_churn_data.csv    # Telco customer dataset
├── frontend/
│   ├── index.html                 # Interactive UI Dashboard
│   ├── style.css                  # Modern UI styles & Glassmorphism design
│   ├── app.js                     # Frontend API interaction logic
│   └── assets/                    # Dashboard icons and visual assets
├── model/
│   ├── churn_model.joblib         # Serialized XGBoost model
│   ├── preprocessor.joblib        # Scikit-learn feature preprocessor
│   └── model_meta.json            # Model parameters, features & evaluation metadata
├── generate_data.py               # Dataset generation & preprocessing script
├── train_model.py                 # Pipeline training, hyperparameter tuning & export script
├── main.py                        # FastAPI Web Server & API Endpoints
└── requirements.txt               # Dependencies list
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Install Dependencies
Clone or open the repository, then install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## 🏁 How to Run

### 1. (Optional) Train the Model
To re-generate synthetic data and retrain the XGBoost model:
```bash
python generate_data.py
python train_model.py
```

### 2. Start the FastAPI Backend
Launch the FastAPI development server:
```bash
uvicorn main:app --reload --port 8000
```
* **API Documentation**: Access Swagger UI at `http://127.0.0.1:8000/docs`

### 3. Open the Frontend Dashboard
Open `frontend/index.html` directly in any web browser, or serve it via a local static server to test live churn predictions against the API.
