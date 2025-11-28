# 06_impute_missing_data.py
import os
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

MISSING_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results', 'missing_data')
OUTPUT_FOLDER = os.path.join(MISSING_FOLDER, 'imputed')

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
