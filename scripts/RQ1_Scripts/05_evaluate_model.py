# 04_evaluate_model.py
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# Set up paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results')
MODEL_PATH = os.path.join(RESULTS_PATH, 'random_forest_rq1.pkl')
X_TEST_PATH = os.path.join(RESULTS_PATH, 'X_test.csv')
Y_TEST_PATH = os.path.join(RESULTS_PATH, 'y_test.csv')

# -----------------------------
# Load test data
# -----------------------------
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

# -----------------------------
# Load trained model
# -----------------------------
rf_model = joblib.load(MODEL_PATH)
print("Random Forest model loaded successfully!\n")

# -----------------------------
# Make predictions
# -----------------------------
y_pred = rf_model.predict(X_test)

# -----------------------------
# Evaluate performance
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-score: {f1:.4f}\n")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=[False, True])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Defective', 'Defective'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix on Test Set")
plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix_test.png'))
plt.close()

print(f"Confusion matrix saved to {RESULTS_PATH}")

# Class-wise performance
from sklearn.metrics import classification_report

report = classification_report(
    y_test, 
    y_pred, 
    target_names=['Non-Defective', 'Defective']
)
print("Class-wise Performance:\n")
print(report)