import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score
)


# Paths Setup
SCRIPT_RESULTS_NAME = "10_evaluate_svm"  # unique folder name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Common results root folder
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Script-specific results folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_ROOT, '09_train_svm', 'svm_rq1.pkl')

# Test data comes from previous data split script
X_TEST_PATH = os.path.join(RESULTS_ROOT, '02_data_split', 'X_test.csv')
Y_TEST_PATH = os.path.join(RESULTS_ROOT, '02_data_split', 'y_test.csv')

# Paths for outputs inside this script-specific folder
REPORT_SAVE_PATH = os.path.join(RESULTS_PATH, 'svm_test_results.txt')
CONF_MATRIX_PLOT = os.path.join(RESULTS_PATH, 'svm_test_confusion_matrix.png')
# Load Model and Test Data

print("Loading test data and SVM model...")
svm_model = joblib.load(MODEL_PATH)

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
print("Data loaded successfully!")



# Predict
print("\nPredicting test data...")
y_pred = svm_model.predict(X_test)
y_prob = svm_model.predict_proba(X_test)[:, 1]  # needed for AUC



# Evaluate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

class_report = classification_report(y_test, y_pred, target_names=["Non-Defective", "Defective"])


print("\n===== SVM Performance on TEST DATA =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("\nClassification Report:")
print(class_report)



# Save Results to txt file
with open(REPORT_SAVE_PATH, "w") as file:
    file.write("SVM Model Evaluation on Test Data\n")
    file.write("-----------------------------------\n")
    file.write(f"Accuracy:  {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall:    {recall:.4f}\n")
    file.write(f"F1-score:  {f1:.4f}\n")
    file.write(f"AUC:       {auc:.4f}\n\n")
    file.write("Classification Report\n")
    file.write(class_report)

print(f"\nSaved evaluation results to {REPORT_SAVE_PATH}")



# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Total samples per class
total_per_class = np.bincount(y_test)  # [total Non-Defective, total Defective]

# Axis labels with totals
classes = ["Non-Defective", "Defective"]
labels_with_totals = [f"{classes[i]} ({total_per_class[i]})" for i in range(len(classes))]

# Cell labels with True/False annotations
cell_text = [
    [f"{cm[0,0]} (True Non-Defective)", f"{cm[0,1]} (False Non-Defective)"],
    [f"{cm[1,0]} (False Defective)", f"{cm[1,1]} (True Defective)"]
]

# Create figure
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
#plt.colorbar(im, ax=ax)

# Set ticks and labels
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(labels_with_totals, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticklabels(labels_with_totals)

# Add cell annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cell_text[i][j], ha="center", va="center", color="black", fontsize=10)

# Axis labels and title
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("SVM Confusion Matrix - Test Data")

plt.tight_layout()
plt.savefig(CONF_MATRIX_PLOT)
plt.show()
plt.close()

print(f"Confusion matrix with totals and descriptive cell labels saved to {CONF_MATRIX_PLOT}")