import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import joblib

# Set modern aesthetic styles for plots to match a premium dark dashboard
plt.style.use('dark_background')
sns.set_palette(['#4ecdc4', '#ff6b6b']) # Teal for No Churn, Coral for Churn
plt.rcParams.update({
    'figure.facecolor': '#1a1a24',
    'axes.facecolor': '#1a1a24',
    'axes.edgecolor': '#2a2a35',
    'grid.color': '#2a2a35',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#cbd5e1',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'font.family': 'sans-serif',
    'figure.autolayout': True
})

def engineer_features(df):
    """
    Perform feature engineering:
    1. Group tenure into categorical buckets.
    2. Count total number of subscribed services.
    3. Calculate MonthlyCharges to TotalCharges ratio.
    """
    # 1. Tenure Grouping
    def get_tenure_group(t):
        if t <= 12: return "0-12 Month"
        elif t <= 24: return "12-24 Month"
        elif t <= 48: return "24-48 Month"
        elif t <= 60: return "48-60 Month"
        else: return "60+ Month"
        
    df['Tenure_Group'] = df['tenure'].apply(get_tenure_group)
    
    # 2. Total Services Count
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]
    df['Total_Services'] = 0
    for col in service_cols:
        if col in df.columns:
            # Add 1 if the service is explicitly 'Yes'
            df['Total_Services'] += (df[col] == 'Yes').astype(int)
            
    # 3. Monthly charges ratio
    # If TotalCharges is 0 (new customer), ratio is MonthlyCharges
    df['Charges_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'].astype(float) + 1e-5)
    
    return df

def main():
    print("Starting Model Training Pipeline...")
    
    # 1. Load Data
    data_path = os.path.join("data", "customer_churn_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Handle missing/empty values in TotalCharges
    # Replace space strings with NaN, then impute with MonthlyCharges * tenure
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    missing_mask = df['TotalCharges'].isna()
    if missing_mask.any():
        df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges'] * df.loc[missing_mask, 'tenure']
        print(f"Imputed {missing_mask.sum()} missing values in TotalCharges.")
        
    # 3. Feature Engineering
    df = engineer_features(df)
    print("Feature engineering completed.")
    
    # Define Target and Features
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Define Columns
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'Tenure_Group'
    ]
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Total_Services', 'Charges_Ratio']
    
    # 4. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples. Test set: {X_test.shape[0]} samples.")
    
    # 5. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Fit preprocessor on training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
    all_features = num_cols + encoded_cat_cols
    
    print(f"Total features after preprocessing: {len(all_features)}")
    
    # 6. Model Training & Hyperparameter Tuning
    print("Tuning XGBoost hyperparameters...")
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.08, 0.15],
        'n_estimators': [100, 150, 200],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=4,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_preprocessed, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # 7. Model Evaluation
    y_pred = best_model.predict(X_test_preprocessed)
    y_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    
    print("\n--- Model Performance ---")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
    print("-------------------------\n")
    
    # 8. Save Model and Preprocessor
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save files
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    joblib.dump(best_model, os.path.join(model_dir, "churn_model.joblib"))
    
    # Save feature names list and metrics metadata
    meta = {
        "metrics": metrics,
        "features": all_features,
        "num_features": num_cols,
        "cat_features": cat_cols,
        "parameters": grid_search.best_params_
    }
    with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"Saved model, preprocessor and metadata to '{model_dir}/'")
    
    # 9. Create Visualizations
    assets_dir = os.path.join("frontend", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Plot 1: Churn by Contract Type (EDA)
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette=['#4ecdc4', '#ff6b6b'])
    plt.title('Customer Churn by Contract Type', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Contract Type', fontsize=11)
    plt.ylabel('Customer Count', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.15)
    plt.savefig(os.path.join(assets_dir, "churn_by_contract.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Tenure vs Monthly Charges Density (EDA)
    plt.figure(figsize=(7, 4.5))
    sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, common_norm=False, palette=['#4ecdc4', '#ff6b6b'], alpha=0.4)
    plt.title('Tenure Distribution by Churn Status', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Tenure (Months)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.grid(linestyle='--', alpha=0.15)
    plt.savefig(os.path.join(assets_dir, "tenure_vs_churn.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Services vs Churn
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=df, x='Total_Services', y='MonthlyCharges', hue='Churn', errorbar=None, palette=['#4ecdc4', '#ff6b6b'])
    plt.title('Monthly Charges & Services by Churn', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Number of Services Subscribed', fontsize=11)
    plt.ylabel('Average Monthly Charges ($)', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.15)
    plt.savefig(os.path.join(assets_dir, "services_vs_churn.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'],
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title('Confusion Matrix', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.savefig(os.path.join(assets_dir, "confusion_matrix.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 5: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#4ecdc4', lw=2.5, label=f'XGBoost (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='#65748b', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curve Analysis', fontsize=14, pad=15, fontweight='bold')
    plt.legend(loc="lower right", facecolor='#1a1a24', edgecolor='#2a2a35')
    plt.grid(linestyle='--', alpha=0.15)
    plt.savefig(os.path.join(assets_dir, "roc_curve.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Feature Importances (Top 10)
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(8, 5))
    # We clean up names slightly for readability
    clean_index = [col.replace('cat__', '').replace('num__', '').replace('_', ' ') for col in feat_imp.index]
    sns.barplot(x=feat_imp.values, y=clean_index, hue=clean_index, legend=False, palette='viridis')
    plt.title('Top 10 Features Driving Churn Risk', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.15)
    plt.savefig(os.path.join(assets_dir, "feature_importance.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    print("All visualization plots saved to frontend/assets/ directory.")
    print("Pipeline executed successfully!")

if __name__ == "__main__":
    main()
