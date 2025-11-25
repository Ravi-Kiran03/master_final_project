# 03_apply_smote.py
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))  # project root
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results')

# Load the preprocessed split datasets
X_train = pd.read_csv(os.path.join(RESULTS_PATH, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(RESULTS_PATH, 'y_train.csv'))

# Flatten y_train if necessary
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

# Plot class distribution after SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_balanced)
plt.title("Class Distribution after SMOTE")
plt.xlabel("Defective (True) vs Non-Defective (False)")
plt.ylabel("Count")
plt.savefig(os.path.join(RESULTS_PATH, "03_class_distribution_after_smote.png"))
plt.close()

print(f"Class distribution plot after SMOTE saved to {RESULTS_PATH}")
