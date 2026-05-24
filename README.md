# Auto Insurance Risk Analytics Engine

An end-to-end data engineering and analytics pipeline built for **AlphaCare Insurance Solutions (ACIS)**. This project focuses on analyzing historical insurance data, performing comprehensive Exploratory Data Analysis (EDA), optimizing pricing/marketing strategies, and identifying low-risk customer segments in the South African market.

---

## 📁 Repository Structure
The workspace is organized following standard data science repository layouts:

```text
auto-insurance-risk-analytics/
├── .dvc/                         # Data Version Control metadata
├── notebooks/                    # Interactive research workspaces
│   └── 01_exploratory_analysis.ipynb
├── src/                          # Modular production source scripts
│   ├── __init__.py
│   ├── data_cleaning.py          # Missing value handling and typing
│   └── hypothesis_testing.py     # Statistical testing suites
├── reports/
│   └── figures/                  # Auto-exported high-res visual assets
│       ├── 01_correlation_matrix.png
│       ├── 02_premium_vs_claim_scatter.png
│       └── 03_risk_outliers_boxplots.png
├── venv/                         # Isolated local Python virtual environment
├── .gitignore                    # System file exclusions (including heavy data files)
├── README.md                     # Project documentation blueprint
└── requirements.txt              # Core package dependencies
```
## 🚀 Environment Setup & Installation Guide

Follow these sequential steps to initialize the environment and run the pipeline inside VS Code:
1. Initialize the Isolated Sandbox Environment

Generate a clean local virtual environment to isolate project packages and prevent global version conflicts:
```bash
   python -m venv venv
```
2. Activate the Environment
```bash
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1

   # On macOS/Linux:
   source venv/bin/activate
```
3. Install Dependencies & Jupyter Core Dependencies

Install the baseline data science toolkits along with ipykernel to bridge the virtual environment directly to your VS Code notebook interface:
```bash
   pip install -r requirements.txt
   pip install ipykernel
```
4. Link the Workspace Kernel to VS Code

   1. Open notebooks/01_exploratory_analysis.ipynb inside VS Code.

   2. Click Select Kernel in the top right-hand corner.

   3. Select Python Environments... -> Choose the interpreter matching .\venv\Scripts\python.exe.

## 📊 Core Analytical Artifacts & Visuals

The following evaluation assets are dynamically generated and archived into reports/figures/ during execution:
1. Continuous Feature Correlation Heatmap

Maps cross-correlations across numerical metrics to pinpoint predictive feature pairs for downstream risk forecasting.
2. Premium Exposure vs. Historical Claim Aggregation

A spatial scatter distribution tracking risk density profiles and total financial exposure, segmented by geographic zones.
3. Risk Variance & Outlier Boxplots

Statistical distributions outlining systemic variance, heavy right-skewed claims distributions, and asset value outliers.

## 📊 Delivered EDA Visuals

* **Correlation:** `![Heatmap](reports/figures/01_correlation_matrix.png)`
* **Risk Scatter:** `![Scatter](reports/figures/02_premium_vs_claim_scatter.png)`
* **Outliers:** `![Boxplots](reports/figures/03_risk_outliers_boxplots.png)`

## 🛠️ Engineering Components

    Data Version Control (DVC): Heavy raw datasets are kept out of Git tracking by leveraging .gitignore and maintaining a secure local cache pointer directory structure.

    Dynamic Attribute Selector: Avoids hardcoded string assumptions. Notebook modules use text-scanning logic to cleanly map source files seamlessly regardless of minor column naming differences.