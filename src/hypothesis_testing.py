import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_hypothesis_testing(df):
    print("🚀 [SYSTEM] Initializing Kifya Task 2 Statistical Engine...\n")
    os.makedirs('../reports/figures', exist_ok=True)
    
    # Dynamic Column Discovery
    premium_col = [c for c in df.columns if 'premium' in c.lower()][0]
    claim_col = [c for c in df.columns if 'claim' in c.lower()][0]
    
    # Look for Gender and Province columns safely
    gender_opts = [c for c in df.columns if 'gender' in c.lower() or 'sex' in c.lower()]
    geo_opts = [c for c in df.columns if c.lower() in ['province', 'region', 'state', 'location']]
    
    # Fallback to categorical columns if exact names aren't found
    gender_col = gender_opts[0] if gender_opts else df.select_dtypes(include=['object', 'category']).columns[0]
    geo_col = geo_opts[0] if geo_opts else df.select_dtypes(include=['object', 'category']).columns[1]

    # --- HYPOTHESIS 1: Total Claims by Gender (Two-Sample t-test) ---
    print(f"📊 Testing H1: Risk differences across {gender_col}...")
    groups = df[gender_col].unique()
    if len(groups) >= 2:
        g1 = df[df[gender_col] == groups[0]][claim_col].dropna()
        g2 = df[df[gender_col] == groups[1]][claim_col].dropna()
        
        t_stat, p_val_t = stats.ttest_ind(g1, g2, equal_var=False)
        print(f"   -> t-statistic: {t_stat:.4f}, p-value: {p_val_t:.4e}")
        print(f"   -> Status: {'🚨 REJECT H0 (Significant)' if p_val_t < 0.05 else '✅ FAIL TO REJECT H0 (No Significant Difference)'}\n")
    
    # --- HYPOTHESIS 2: Claim Frequency / Risk Profile across Provinces (Chi-Square Test) ---
    print(f"📊 Testing H2: Claim distribution independence across {geo_col}...")
    # Create a high vs low claim flag to build a contingency table
    median_claim = df[claim_col].median()
    df['High_Claim_Risk'] = df[claim_col].apply(lambda x: 'High' if x > median_claim else 'Low')
    
    contingency_table = pd.crosstab(df[geo_col], df['High_Claim_Risk'])
    chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"   -> Chi-Square: {chi2:.4f}, p-value: {p_val_chi2:.4e}")
    print(f"   -> Status: {'🚨 REJECT H0 (Significant)' if p_val_chi2 < 0.05 else '✅ FAIL TO REJECT H0'}\n")
    
    # --- VISUALIZATION: Save Statistical Proofs ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x=gender_col, y=claim_col, estimator=np.mean, errorbar='ci', palette='muted')
    plt.title(f'Mean Historical Claims by {gender_col}')
    plt.ylabel('Mean Claims (ZAR)')
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x=geo_col, hue='High_Claim_Risk', palette='viridis')
    plt.title(f'Risk Segmentation across {geo_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = '../reports/figures/04_hypothesis_testing_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Exported statistical charts to: {output_path}")
    plt.show()

# To execute directly in notebook run:
# from src.hypothesis_testing import run_hypothesis_testing
# run_hypothesis_testing(df)