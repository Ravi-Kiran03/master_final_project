# 05_create_missing_data.py
import os
import pandas as pd
import numpy as np

# Set up paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results', 'X_test.csv')
MISSING_FOLDER = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results', 'missing_data')

os.makedirs(MISSING_FOLDER, exist_ok=True)


X_test = pd.read_csv(DATA_PATH)

# Function to introduce MCAR missingness

def introduce_mcar(df, missing_pct, random_state=42):
    """
    Introduces Missing Completely At Random (MCAR) in the dataset.
    
    :param df: input DataFrame
    :param missing_pct: fraction of values to set as NaN (0-100)
    :param random_state: random seed
    :return: DataFrame with missing values
    """
    np.random.seed(random_state)
    df_missing = df.copy()
    total_values = df_missing.size
    n_missing = int(total_values * (missing_pct / 100))
    
    # Randomly choose indices for missing values
    missing_indices = (
        np.random.choice(total_values, n_missing, replace=False)
    )
    
    # Convert 1D indices to 2D indices for DataFrame
    rows, cols = np.unravel_index(missing_indices, df_missing.shape)
    df_missing.values[rows, cols] = np.nan
    
    return df_missing


# Generate missing datasets at 10%, 20%, 30%, 40%

missing_levels = [10, 20, 30, 40]

for pct in missing_levels:
    df_missing = introduce_mcar(X_test, pct)
    save_path = os.path.join(MISSING_FOLDER, f'X_test_missing_{pct}pct.csv')
    df_missing.to_csv(save_path, index=False)
    print(f"Saved {pct}% missing data to {save_path}")
