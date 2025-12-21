# 07_impute_missing_data.py
import os
import pandas as pd
from sklearn.impute import SimpleImputer

# Script-specific results folder
SCRIPT_RESULTS_NAME = "07_impute_missing_data"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Common results root folder
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Script-specific results folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Missing datasets are loaded from previous script folder
MISSING_FOLDER = os.path.join(RESULTS_ROOT, '06_create_missing_data')

# Output folder for imputed datasets (inside this script's folder)
OUTPUT_FOLDER = RESULTS_PATH
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

missing_levels = [10, 20, 30, 40]

imputer = SimpleImputer(strategy='mean')

for pct in missing_levels:
    file_path = os.path.join(MISSING_FOLDER, f"X_test_missing_{pct}pct.csv")
    df_missing = pd.read_csv(file_path)

    # Apply mean imputation
    df_imputed = pd.DataFrame(imputer.fit_transform(df_missing),
                              columns=df_missing.columns)

    # Save imputed file
    save_path = os.path.join(OUTPUT_FOLDER, f"X_test_missing_{pct}pct_imputed.csv")
    df_imputed.to_csv(save_path, index=False)

    print(f"Saved imputed dataset: {save_path}")
