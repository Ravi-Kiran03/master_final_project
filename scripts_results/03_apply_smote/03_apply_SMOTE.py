# 03_apply_smote.py
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_RESULTS_NAME = "03_apply_smote"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Common results root folder
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Script-specific results folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load the preprocessed split datasets from previous script folder
X_train = pd.read_csv(os.path.join(RESULTS_ROOT, '02_data_split', 'X_train.csv'))
y_train = pd.read_csv(os.path.join(RESULTS_ROOT, '02_data_split', 'y_train.csv'))
y_train = y_train.values.ravel()


print("Original training set class distribution:")
print(pd.Series(y_train).value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE, class distribution:")
print(pd.Series(y_train_balanced).value_counts())

# Save the balanced dataset
pd.DataFrame(X_train_balanced, columns=X_train.columns).to_csv(
    os.path.join(RESULTS_PATH, 'X_train_balanced.csv'), index=False
)
pd.DataFrame(y_train_balanced, columns=['defects']).to_csv(
    os.path.join(RESULTS_PATH, 'y_train_balanced.csv'), index=False
)

print(f"\nBalanced training set saved to {RESULTS_PATH}")

plt.figure(figsize=(6,4))
ax = sns.countplot(x=y_train_balanced)
plt.title("Class Distribution after SMOTE")
ax.set_xticklabels(['Non-Defective', 'Defective'])

total = len(y_train_balanced)
for p in ax.patches:
    height = p.get_height()
    percentage = height / total * 100
    ax.annotate(f'{height} ({percentage:.1f}%)',
                (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "03_class_distribution_after_smote.png"))
plt.close()
