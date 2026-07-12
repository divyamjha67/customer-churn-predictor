from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import json

app = FastAPI(title="Customer Churn Prediction System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory for static assets
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Paths for the saved models and preprocessors
MODEL_PATH = os.path.join("model", "churn_model.joblib")
PREPROCESSOR_PATH = os.path.join("model", "preprocessor.joblib")
META_PATH = os.path.join("model", "model_meta.json")

# Loaded objects
model = None
preprocessor = None
model_meta = None

@app.on_event("startup")
def load_assets():
    global model, preprocessor, model_meta
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("Model and Preprocessor pipeline loaded successfully.")
        else:
            print("Warning: Model or Preprocessor file not found. Ensure train_model.py has run successfully.")
            
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                model_meta = json.load(f)
            print("Model metadata loaded successfully.")
    except Exception as e:
        print(f"Error loading assets: {str(e)}")

# Request Pydantic Model
class CustomerDataRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Helper to engineer features on a single row/dataframe
def engineer_features(df):
    # Tenure buckets
    def get_tenure_group(t):
        if t <= 12: return "0-12 Month"
        elif t <= 24: return "12-24 Month"
        elif t <= 48: return "24-48 Month"
        elif t <= 60: return "48-60 Month"
        else: return "60+ Month"
        
    df['Tenure_Group'] = df['tenure'].apply(get_tenure_group)
    
    # Total services count
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]
    df['Total_Services'] = 0
    for col in service_cols:
        if col in df.columns:
            df['Total_Services'] += (df[col] == 'Yes').astype(int)
            
    # Charges ratio
    df['Charges_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'].astype(float) + 1e-5)
    return df

@app.post("/api/predict")
def predict_churn(request: CustomerDataRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="ML model pipeline is not loaded on the server.")
        
    try:
        # Convert Pydantic request to pandas DataFrame (matching keys)
        input_data = request.dict()
        df_input = pd.DataFrame([input_data])
        
        # Run feature engineering
        df_engineered = engineer_features(df_input)
        
        # Preprocess features
        features_preprocessed = preprocessor.transform(df_engineered)
        
        # Run prediction
        proba = float(model.predict_proba(features_preprocessed)[0][1])
        prediction_val = int(model.predict(features_preprocessed)[0])
        prediction_label = "Yes" if prediction_val == 1 or proba >= 0.50 else "No"
        
        # Categorize risk levels
        if proba < 0.30:
            risk_level = "Low Risk"
        elif proba < 0.65:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        # Explanations & recommendations based on rules
        factors = []
        recommendations = []
        
        if request.Contract == "Month-to-month":
            factors.append("Contract type: Month-to-month contracts lack long-term commitments and show the highest probability of churn.")
            recommendations.append("Offer a promotional 10-15% discount for upgrading to a 1-year or 2-year contract.")
            
        if request.InternetService == "Fiber optic":
            factors.append("Internet Service: Fiber optic customers have higher premium costs, resulting in higher churn frequencies.")
            recommendations.append("Validate fiber installation quality or review value adding bundles (e.g. streaming discounts).")
            
        if request.OnlineSecurity == "No" or request.TechSupport == "No":
            factors.append("Missing Addons: Customer lacks security or tech support addons which decreases overall account stickiness.")
            recommendations.append("Offer a free trial or discounted bundle for Online Security and Tech Support services.")
            
        if request.PaymentMethod == "Electronic check":
            factors.append("Payment Method: Manual monthly billing via electronic check is highly correlated with elevated churn rates.")
            recommendations.append("Offer a one-time bill credit (e.g. $5) for setting up automatic Credit Card or Bank Transfer payments.")
            
        if request.MonthlyCharges > 85.0:
            factors.append("High Cost: High monthly charges exceeding $85 increase cost sensitivity.")
            recommendations.append("Review billing plans to identify if they can be transitioned to cheaper bundles with lower pricing.")
            
        if request.tenure < 12:
            factors.append("Low Tenure: Customers in their first year are in the highest churn hazard window.")
            recommendations.append("Schedule a proactive customer success touchpoint or send a loyalty engagement reward.")
            
        # Default factors/rec if empty
        if not factors:
            factors.append("Customer shows stable usage and tenure patterns with low risk indicators.")
            recommendations.append("Maintain standard service quality. Continue standard loyalty programs.")
            
        return {
            "prediction": prediction_label,
            "probability": proba,
            "risk_level": risk_level,
            "key_factors": factors,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/metrics")
def get_metrics():
    if model_meta is None:
        raise HTTPException(status_code=500, detail="Metadata is not loaded on the server.")
    return model_meta

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
