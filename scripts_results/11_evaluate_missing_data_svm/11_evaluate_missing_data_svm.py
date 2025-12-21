# 08_evaluate_missing_data_svm.py

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Paths
SCRIPT_RESULTS_NAME = "11_evaluate_missing_data_svm"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Common results root folder
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Script-specific folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)


# Paths for model and data
MODEL_PATH = os.path.join(RESULTS_ROOT, '09_train_svm', 'svm_rq1.pkl')  # change if needed
X_TEST_PATH = os.path.join(RESULTS_ROOT, '02_data_split', 'X_test.csv')
Y_TEST_PATH = os.path.join(RESULTS_ROOT, '02_data_split', 'y_test.csv')

IMPUTED_FOLDER = os.path.join(os.path.dirname(RESULTS_PATH), '07_impute_missing_data')
OUTPUT_CSV = os.path.join(RESULTS_PATH, 'missingness_results_svm.csv')
PLOT_PATH = os.path.join(RESULTS_PATH, 'missingness_performance_svm.png')


# Load model and test labels
model = joblib.load(MODEL_PATH)
X_test_original = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()


# Evaluate different missingness levels
missing_levels = [10, 20, 30, 40]
results = []

for pct in missing_levels:
    imputed_file = os.path.join(IMPUTED_FOLDER, f"X_test_missing_{pct}pct_imputed.csv")
    X_test_imputed = pd.read_csv(imputed_file)

    # Predict using SVM model
    y_pred = model.predict(X_test_imputed)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results.append({
        "missingness_pct": pct,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    })

    print(f"Missingness {pct}% → Accuracy: {accuracy:.4f}, F1: {f1:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}")


# Save results into CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved results to:")
print(OUTPUT_CSV)

# Plot performance vs missingness
plt.figure(figsize=(8,5))
plt.plot(results_df['missingness_pct'], results_df['accuracy'], marker='o', label='Accuracy')
plt.plot(results_df['missingness_pct'], results_df['f1_score'], marker='o', label='F1 Score')
plt.plot(results_df['missingness_pct'], results_df['precision'], marker='o', label='Precision')
plt.plot(results_df['missingness_pct'], results_df['recall'], marker='o', label='Recall')
plt.xlabel('Missingness (%)')
plt.ylabel('Score')
plt.title('SVM Performance vs Missingness')
plt.xticks(missing_levels)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print(f"Performance plot saved to: {PLOT_PATH}")
