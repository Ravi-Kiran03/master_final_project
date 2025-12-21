# 02_data_split.py - Split dataset + scale features
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_RESULTS_NAME = "02_data_split"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'kc1_raw.csv')
# Results root folder (common for all scripts)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')  # or 'results'

# Script-specific results folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Basic check
if df.isnull().sum().sum() > 0:
    print("Warning: Dataset contains missing values!")

# Features/target
X = df.drop(columns=['defects'])
y = df['defects']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Class distribution check
print("\nClass distribution in full dataset:")
print(y.value_counts())

# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nClass distribution in training set:")
print(y_train.value_counts())

print("\nClass distribution in testing set:")
print(y_test.value_counts())

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaled datasets
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    os.path.join(RESULTS_PATH, 'X_train.csv'), index=False
)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    os.path.join(RESULTS_PATH, 'X_test.csv'), index=False
)
y_train.to_csv(os.path.join(RESULTS_PATH, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(RESULTS_PATH, 'y_test.csv'), index=False)

# Save the scaler
joblib.dump(scaler, os.path.join(RESULTS_PATH, 'scaler.pkl'))

print(f"\nSaved to {RESULTS_PATH}")


# Plot class distribution in training set with counts + percentages
plt.figure(figsize=(6,4))
ax = sns.countplot(x=y_train)

# Title
plt.title("Training Set Class Distribution Before SMOTE")

# Rename x-axis labels
ax.set_xticklabels(['Non-Defective', 'Defective'])

# Add counts + percentages on top of bars
total = len(y_train)
for p in ax.patches:
    height = p.get_height()
    percentage = height / total * 100
    ax.annotate(f'{height} ({percentage:.1f}%)',
                (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "02_class_distribution_train_set.png"))
plt.close()


# Plot class distribution in testing set with counts + percentages
plt.figure(figsize=(6,4))
ax = sns.countplot(x=y_test)

# Title
plt.title("Testing Set Class Distribution")

# Rename x-axis labels
ax.set_xticklabels(['Non-Defective', 'Defective'])

# Add counts + percentages on top of bars
total = len(y_test)
for p in ax.patches:
    height = p.get_height()
    percentage = height / total * 100
    ax.annotate(f'{height} ({percentage:.1f}%)',
                (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "02_class_distribution_test_set.png"))
plt.close()
