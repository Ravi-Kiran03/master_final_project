# 02_data_split.py - Split dataset + scale features
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'kc1_raw.csv')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results')
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
    X, y, test_size=0.3, random_state=42, stratify=y
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


# Plot class distribution in training set
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Training Set Class Distribution Before SMOTE")
plt.xlabel("Defective (True) vs Non-Defective (False)")
plt.ylabel("Count")
plt.savefig(os.path.join(RESULTS_PATH, "02_class_distribution_train_set.png"))
plt.close()

# Plot class distribution in testing set
plt.figure(figsize=(6,4))
sns.countplot(x=y_test)
plt.title("Testing Set Class Distribution")
plt.xlabel("Defective (True) vs Non-Defective (False)")
plt.ylabel("Count")
plt.savefig(os.path.join(RESULTS_PATH, "02_class_distribution_test_set.png"))
plt.close()
