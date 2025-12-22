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
OUTPUT_CSV = os.path.join(RESULTS_PATH, 'comparison_missingness_table.csv')


# Load result files

df_rf = pd.read_csv(RF_RESULTS)
df_svm = pd.read_csv(SVM_RESULTS)

missing_levels = df_rf['missingness_pct']

#CSV

comparison_data = []

for i, pct in enumerate(missing_levels):
    row = {
        "Missingness (%)": pct,
        "RF Accuracy": df_rf.loc[i, 'accuracy'],
        "SVM Accuracy": df_svm.loc[i, 'accuracy'],
        "RF F1": df_rf.loc[i, 'f1_score'],
        "SVM F1": df_svm.loc[i, 'f1_score'],
        "RF Precision": df_rf.loc[i, 'precision'],
        "SVM Precision": df_svm.loc[i, 'precision'],
        "RF Recall": df_rf.loc[i, 'recall'],
        "SVM Recall": df_svm.loc[i, 'recall'],
    }

    desc = []
    if df_rf.loc[i, 'accuracy'] > df_svm.loc[i, 'accuracy']:
        desc.append("RF more robust")
    elif df_rf.loc[i, 'accuracy'] < df_svm.loc[i, 'accuracy']:
        desc.append("SVM more robust")
    else:
        desc.append("Similar accuracy")

    if df_rf.loc[i, 'f1_score'] > df_svm.loc[i, 'f1_score']:
        desc.append("RF better F1")
    elif df_rf.loc[i, 'f1_score'] < df_svm.loc[i, 'f1_score']:
        desc.append("SVM better F1")

    if pct >= 30:
        desc.append("Performance drops at high missingness")

    row["Description"] = "; ".join(desc)
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)
df_comparison.to_csv(OUTPUT_CSV, index=False)

print("\nComparison table saved:")
print(OUTPUT_CSV)


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
