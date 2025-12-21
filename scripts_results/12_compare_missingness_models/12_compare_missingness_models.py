# 12_compare_missingness_models.py

import os
import pandas as pd
import matplotlib.pyplot as plt


# Script-specific results folder
SCRIPT_RESULTS_NAME = "12_compare_missingness_models"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)


# Input CSVs from previous scripts
RF_RESULTS = os.path.join(RESULTS_ROOT, '08_evaluate_missing_data', 'missingness_results.csv')
SVM_RESULTS = os.path.join(RESULTS_ROOT, '11_evaluate_missing_data_svm', 'missingness_results_svm.csv')

OUTPUT_PLOT = os.path.join(RESULTS_PATH, 'comparison_missingness_performance_full.png')


# Load result files

df_rf = pd.read_csv(RF_RESULTS)
df_svm = pd.read_csv(SVM_RESULTS)

missing_levels = df_rf['missingness_pct']


# Plot

plt.figure(figsize=(10, 7))

# ---- Accuracy ----
plt.plot(missing_levels, df_rf['accuracy'], marker='o', linestyle='-', label='RF Accuracy')
plt.plot(missing_levels, df_svm['accuracy'], marker='o', linestyle='--', label='SVM Accuracy')

# ---- F1 ----
plt.plot(missing_levels, df_rf['f1_score'], marker='s', linestyle='-', label='RF F1')
plt.plot(missing_levels, df_svm['f1_score'], marker='s', linestyle='--', label='SVM F1')

# ---- Precision ----
plt.plot(missing_levels, df_rf['precision'], marker='^', linestyle='-', label='RF Precision')
plt.plot(missing_levels, df_svm['precision'], marker='^', linestyle='--', label='SVM Precision')

# ---- Recall ----
plt.plot(missing_levels, df_rf['recall'], marker='d', linestyle='-', label='RF Recall')
plt.plot(missing_levels, df_svm['recall'], marker='d', linestyle='--', label='SVM Recall')

plt.xlabel("Missingness (%)")
plt.ylabel("Score")
plt.title("Full Model Comparison vs Missingness (RF vs SVM)")
plt.xticks(missing_levels)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_PLOT)
plt.close()

print('\nComparison plot saved:')
print(OUTPUT_PLOT)
