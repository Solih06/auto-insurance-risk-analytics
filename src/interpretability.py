import os
import shap
import matplotlib.pyplot as plt

def generate_model_explanations(model, df, features):
    """
    Task 5: Builds SHAP interpretability structures to map how core variables
    systematically push underwriting pricing recommendations above or below baselines.
    """
    print("\n🚀 [SYSTEM] Executing Model Interpretability Framework (SHAP Engine)...")
    os.makedirs('../reports/figures', exist_ok=True)
    
    # Isolate training subset feature array
    X = df[features]
    
    # Initialize TreeExplainer for boosted ensembles
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # --- VISUALIZATION: Save Feature Impact Summary ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Impact Directionality Diagnostics", fontsize=14, pad=15)
    
    output_path = '../reports/figures/05_shap_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Exported SHAP explainability charts to: {output_path}")
    plt.close()