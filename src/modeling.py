import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, log_loss, accuracy_score

def prepare_modeling_features(df):
    """
    Dynamically identifies and processes variables for statistical modeling, 
    preserving structural consistency with the data engineering loop.
    """
    df_clean = df.copy()
    
    # 1. Dynamic Attribute Mapping (Same convention as Task 2 script)
    premium_col = [c for c in df_clean.columns if 'premium' in c.lower()][0]
    claim_col = [c for c in df_clean.columns if 'claim' in c.lower()][0]
    
    # Feature Engineering Layer: Convert Registration Years to an active Age metric
    reg_year_opts = [c for c in df_clean.columns if 'year' in c.lower() or 'reg' in c.lower()]
    if reg_year_opts:
        df_clean['VehicleAge'] = 2026 - df_clean[reg_year_opts[0]]
    else:
        df_clean['VehicleAge'] = 0
        
    # Isolate numeric continuous features to act as base predictors
    numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    # Safely discard direct target labels from the training domain feature space
    for col in [premium_col, claim_col, 'PolicyID']:
        if col in numeric_features:
            numeric_features.remove(col)
            
    return df_clean, numeric_features, claim_col

def train_risk_models(df):
    """
    Trains and benchmarks regressions (Severity) and classification (Probability) 
    models to establish a dynamic premium computation foundation.
    """
    print("🚀 [SYSTEM] Initializing Task 4 Statistical Modeling Loop...\n")
    df_proc, features, claim_col = prepare_modeling_features(df)
    
    # --- MODELING PART 1: CLAIM SEVERITY MODELING (REGRESSION) ---
    print(f"📊 Running Severity Benchmarking (Subset where {claim_col} > 0)...")
    severity_df = df_proc[df_proc[claim_col] > 0]
    
    if len(severity_df) > 10:
        X_sev = severity_df[features]
        y_sev = severity_df[claim_col]
        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_sev, y_sev, test_size=0.2, random_state=42)
        
        reg_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        }
        
        for name, model in reg_models.items():
            model.fit(X_tr_s, y_tr_s)
            preds = model.predict(X_te_s)
            print(f"   -> {name} | RMSE: {np.sqrt(mean_squared_error(y_te_s, preds)):.2f}, R2 Score: {r2_score(y_te_s, preds):.4f}")
    else:
        print("   -> [WARNING] Insufficient data samples containing incurred historical losses for regression metrics.")
        reg_models = {"XGBoost Regressor": XGBRegressor().fit(df_proc[features], df_proc[claim_col])}

    # --- MODELING PART 2: CLAIM PROBABILITY MODELING (BINARY CLASSIFICATION) ---
    print(f"\n📊 Running Probability Benchmarking...")
    df_proc['Claim_Occurred'] = (df_proc[claim_col] > 0).astype(int)
    
    X_prob = df_proc[features]
    y_prob = df_proc['Claim_Occurred']
    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_prob, y_prob, test_size=0.2, random_state=42)
    
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in clf_models.items():
        model.fit(X_tr_p, y_tr_p)
        prob_preds = model.predict_proba(X_te_p)[:, 1]
        print(f"   -> {name} | LogLoss Score: {log_loss(y_te_p, prob_preds):.4f}, Accuracy: {accuracy_score(y_te_p, model.predict(X_te_p)):.4f}")
        
    return reg_models["XGBoost Regressor"], clf_models["XGBoost Classifier"], features

def calculate_risk_premium(df, severity_model, probability_model, features, expense_loading=150.0, profit_margin=0.15):
    """
    Task 4.3: Structural Premium Optimization Formula Integration.
    Premium = (P(Claim) * Predicted Severity + Expense Loading) / (1 - Profit Margin)
    """
    df_proc, _, _ = prepare_modeling_features(df)
    X = df_proc[features]
    
    # Calculate probability vectors and expected loss magnitudes
    p_claim = probability_model.predict_proba(X)[:, 1]
    pred_severity = severity_model.predict(X)
    
    pure_premium = p_claim * pred_severity
    optimized_premiums = (pure_premium + expense_loading) / (1 - profit_margin)
    
    df_proc['Calculated_Risk_Premium'] = optimized_premiums
    print("\n✅ [SUCCESS] Risk-based Premium engine optimized across portfolio variables.")
    return df_proc