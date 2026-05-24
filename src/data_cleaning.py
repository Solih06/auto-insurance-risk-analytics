import os
import pandas as pd
import numpy as np

class AutoInsurancePipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        """Loads data from raw DVC-managed folder."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Source file missing at: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        print(f"[INFO] Initial Ingestion Complete. Base dimensions: {self.df.shape}")
        return self.df

    def handle_missing_values(self):
        """Systematically cleans missing fields based on standard domain practices."""
        print("[INFO] Auditing missing value thresholds...")
        
        # Numeric columns get median imputation to protect against skewness
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f" Imputed missing numbers in '{col}' with median: {median_val}")
                
        # Categorical columns get mode/unspecified fallback imputation
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unspecified'
                self.df[col].fillna(mode_val, inplace=True)
                print(f" Imputed missing categories in '{col}' with mode: {mode_val}")

    def treat_outliers_iqr(self, target_column):
        """Flags extreme values using the 1.5 * IQR standard boundary."""
        if target_column not in self.df.columns:
            return
            
        q1 = self.df[target_column].quantile(0.25)
        q3 = self.df[target_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Instead of throwing out real historic high claims, cap them or flag them
        outliers_count = ((self.df[target_column] < lower_bound) | (self.df[target_column] > upper_bound)).sum()
        print(f"[OUTLIER AUDIT] '{target_column}' upper limit: {upper_bound:.2f}. Identified records beyond limit: {outliers_count}")

    def save_processed_state(self):
        """Saves a dense clean baseline ready for analytical modeling modeling."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        print(f"[SUCCESS] Exported processed dataset to: {self.output_path}. Dimensions: {self.df.shape}\n")

if __name__ == "__main__":
    # Define production lifecycle parameters
    INPUT_FILE = "data/raw/insurance_data.csv"
    OUTPUT_FILE = "data/processed/cleaned_insurance_data.csv"
    
    pipeline = AutoInsurancePipeline(input_path=INPUT_FILE, output_path=OUTPUT_FILE)
    
    try:
        pipeline.load_data()
        pipeline.handle_missing_values()
        
        # Run outlier tracking loops on targets
        pipeline.treat_outliers_iqr('TotalPremium')
        pipeline.treat_outliers_iqr('TotalClaim')
        
        pipeline.save_processed_state()
        
    except Exception as e:
        print(f"[FATAL FAILURE] Pipeline run aborted: {e}")