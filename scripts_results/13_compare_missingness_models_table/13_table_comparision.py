# 13_compare_missingness_models_table.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Script-specific results folder
# -----------------------------
SCRIPT_RESULTS_NAME = "13_compare_missingness_models_table"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

# -----------------------------
# Input CSVs
# -----------------------------
RF_RESULTS = os.path.join(RESULTS_ROOT, '08_evaluate_missing_data', 'missingness_results.csv')
SVM_RESULTS = os.path.join(RESULTS_ROOT, '11_evaluate_missing_data_svm', 'missingness_results_svm.csv')

OUTPUT_CSV = os.path.join(RESULTS_PATH, 'comparison_missingness_table.csv')
OUTPUT_IMG = os.path.join(RESULTS_PATH, 'comparison_missingness_table.png')

# -----------------------------
# Load CSVs
# -----------------------------
df_rf = pd.read_csv(RF_RESULTS)
df_svm = pd.read_csv(SVM_RESULTS)

# -----------------------------
# Create comparison DataFrame
# -----------------------------
comparison_data = []

for i, pct in enumerate(df_rf['missingness_pct']):
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

    # Add description
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

# -----------------------------
# Save to CSV
# -----------------------------
df_comparison.to_csv(OUTPUT_CSV, index=False)
print(f"Comparison table saved to CSV:\n{OUTPUT_CSV}")

# -----------------------------
# Save as clean image
# -----------------------------
fig, ax = plt.subplots(figsize=(16, len(df_comparison)*0.6 + 2))
ax.axis('off')  # Hide axes

# Render table
table = ax.table(cellText=df_comparison.values,
                 colLabels=df_comparison.columns,
                 cellLoc='center',
                 loc='center')

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # adjust row height

plt.title("Comparison of RF vs SVM Performance vs Missingness", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
plt.close()

print(f"Comparison table saved as image:\n{OUTPUT_IMG}")
