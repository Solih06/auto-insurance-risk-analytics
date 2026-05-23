import pandas as pd
import numpy as np
import os

class InsuranceDataEngine:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Loads dataset safely and profiles its shape."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Target data file not found at: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        print(f"[INFO] Dataset loaded successfully. Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        return self.df

    def evaluate_missing_data(self):
        """Calculates exact missing value distributions across columns."""
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_count,
            'Percentage_(%)': missing_pct
        }).sort_values(by='Percentage_(%)', ascending=False)
        return missing_df[missing_df['Missing_Count'] > 0]

    def profile_numerical_distributions(self):
        """Returns foundational statistical descriptions for key variables."""
        target_cols = ['TotalPremium', 'TotalClaim', 'SumInsured']
        # Filter down only to existing target columns to avoid unexpected KeyErrors
        valid_cols = [col for col in target_cols if col in self.df.columns]
        return self.df[valid_cols].describe()

    def identify_low_risk_segments(self, dimension_col):
        """
        Groups data by a demographic or geographic dimension to evaluate 
        risk optimization options (Claim-to-Premium Ratio).
        """
        if dimension_col not in self.df.columns:
            return f"Dimension column '{dimension_col}' does not exist."
            
        # Group and aggregate risk metrics
        risk_matrix = self.df.groupby(dimension_col).agg(
            Total_Premium=('TotalPremium', 'sum'),
            Total_Claims=('TotalClaim', 'sum'),
            Average_Claim=('TotalClaim', 'mean'),
            Exposure_Count=(dimension_col, 'count')
        ).reset_index()
        
        # Calculate Risk Margin Ratio (Lower means higher profitability / lower risk)
        risk_matrix['Claim_to_Premium_Ratio'] = risk_matrix['Total_Claims'] / risk_matrix['Total_Premium']
        return risk_matrix.sort_values(by='Claim_to_Premium_Ratio', ascending=True)

if __name__ == "__main__":
    # Update this path to where your local dataset file lives inside data/raw/
    DATA_PATH = "data/raw/insurance_data.csv" 
    
    try:
        engine = InsuranceDataEngine(file_path=DATA_PATH)
        df = engine.load_data()
        
        print("\n=== 1. MISSING DATA PROFILING ===")
        missing_summary = engine.evaluate_missing_data()
        print(missing_summary if not missing_summary.empty else "No missing values found across features!")
        
        print("\n=== 2. NUMERICAL DESCRIPTIVE STATISTICS ===")
        print(engine.profile_numerical_distributions())
        
        # Testing geography-based risk insights (e.g., Province)
        if 'Province' in df.columns:
            print("\n=== 3. REGIONAL RISK MARGIN PROFILING (Sorted Low to High Risk) ===")
            print(engine.identify_low_risk_segments(dimension_col='Province'))
            
    except Exception as e:
        print(f"[ERROR] Engine runtime failure: {e}")