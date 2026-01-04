# 14_lime_explanations.py
# LIME explanations for Random Forest model (RQ2)

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# Script-specific results folder
# -----------------------------
SCRIPT_RESULTS_NAME = "14_lime_explanations"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Common results root
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# This script's output folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

# -----------------------------
# Inputs from previous scripts
# -----------------------------
MODEL_PATH = os.path.join(
    RESULTS_ROOT,
    '04_train_random_forest',
    'random_forest_rq1.pkl'
)

X_TRAIN_PATH = os.path.join(
    RESULTS_ROOT,
    '02_data_split',
    'X_train.csv'
)

X_TEST_PATH = os.path.join(
    RESULTS_ROOT,
    '02_data_split',
    'X_test.csv'
)

Y_TEST_PATH = os.path.join(
    RESULTS_ROOT,
    '02_data_split',
    'y_test.csv'
)

# -----------------------------
# Load model and data
# -----------------------------
print("Loading model and data...")

rf_model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print("Model and data loaded successfully!\n")

# -----------------------------
# Initialize LIME explainer
# -----------------------------
print("Initializing LIME explainer...")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=["Non-Defective", "Defective"],
    mode="classification",
    discretize_continuous=True
)

print("LIME explainer initialized!\n")

# -----------------------------
# Explain one defective instance
# -----------------------------
print("Generating LIME explanation...")

# Pick first defective instance
positive_index = list(y_test).index(1)
instance = X_test.iloc[positive_index].values

exp = lime_explainer.explain_instance(
    data_row=instance,
    predict_fn=rf_model.predict_proba,
    num_features=10
)

# -----------------------------
# Save explanation as plot
# -----------------------------
# Extract feature names and contributions
feature_names = [f[0] for f in exp.as_list()]
contributions = [f[1] for f in exp.as_list()]

# Color bars by direction
colors = ['green' if val > 0 else 'red' for val in contributions]

plt.figure(figsize=(8,5))
plt.barh(feature_names, contributions, color=colors)
plt.xlabel("Feature Contribution to Prediction")
plt.title("Local Feature Contributions for Defective Module (LIME)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "lime_defective_instance.png"))
plt.close()

# -----------------------------
# Save explanation as text
# -----------------------------
with open(os.path.join(RESULTS_PATH, "lime_defective_instance.txt"), "w") as f:
    f.write(exp.as_list().__str__())

print("\nLIME explanation saved successfully to:")
print(RESULTS_PATH)
