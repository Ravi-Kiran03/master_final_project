# 07_evaluate_missing_data.py

import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results')

MODEL_PATH = os.path.join(RESULTS_PATH, 'random_forest_rq1.pkl')

X_TEST_PATH = os.path.join(RESULTS_PATH, 'X_test.csv')
Y_TEST_PATH = os.path.join(RESULTS_PATH, 'y_test.csv')

IMPUTED_FOLDER = os.path.join(RESULTS_PATH, 'missing_data', 'imputed')
OUTPUT_CSV = os.path.join(RESULTS_PATH, 'missing_data', 'missingness_results.csv')


model = joblib.load(MODEL_PATH)
X_test_original = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

# Evaluate different missingness levels

missing_levels = [10, 20, 30, 40]
results = []

for pct in missing_levels:
    imputed_file = os.path.join(IMPUTED_FOLDER, f"X_test_missing_{pct}pct_imputed.csv")
    X_test_imputed = pd.read_csv(imputed_file)

    # Predict
    y_pred = model.predict(X_test_imputed)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "missingness_pct": pct,
        "accuracy": accuracy,
        "f1_score": f1
    })

    print(f"Missingness {pct}% → Accuracy: {accuracy:.4f}, F1: {f1:.4f}")



results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved results to:")
print(OUTPUT_CSV)
