import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt


# Script-specific results folder
SCRIPT_RESULTS_NAME = "05_evaluate_random_forest"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load model and test data
MODEL_PATH = os.path.join(RESULTS_ROOT, '04_train_random_forest', 'random_forest_rq1.pkl')
X_TEST_PATH = os.path.join(RESULTS_ROOT, '02_data_split', 'X_test.csv')
Y_TEST_PATH = os.path.join(RESULTS_ROOT, '02_data_split', 'y_test.csv')

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!\n")



# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # For AUC & ROC


# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n===== Model Evaluation Results on TEST DATA =====\n")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}\n")


# Save metrics to file
metrics_output_path = os.path.join(RESULTS_PATH, "evaluation_metrics.txt")
with open(metrics_output_path, "w") as f:
    f.write("Model Performance on Test Data\n")
    f.write("-----------------------------------\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")
    f.write(f"AUC:       {auc:.4f}\n")

print(f"Performance metrics saved to {metrics_output_path}")


# Classification Report
report = classification_report(y_test, y_pred, target_names=["Non-Defective", "Defective"])

report_path = os.path.join(RESULTS_PATH, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

print(f"Classification report saved to {report_path}\n")


# Confusion Matrix Plot
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Count total samples per class
unique, counts = np.unique(y_test, return_counts=True)
class_totals = dict(zip(unique, counts))  # {0: 357, 1: 65}

# Axis labels with totals
labels_with_totals = [
    f"Non-Defective ({class_totals[0]})",
    f"Defective ({class_totals[1]})"
]

# Cell text labels
cell_text = [
    [f"{cm[0,0]} (True Non-Defective)", f"{cm[0,1]} (False Non-Defective)"],
    [f"{cm[1,0]} (False Defective)", f"{cm[1,1]} (True Defective)"]
]

# Plot
fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(cm, cmap='Blues')

# Title
ax.set_title("Random Forest Confusion Matrix - Test Set", fontsize=14)

# Tick labels
ax.set_xticks(np.arange(len(labels_with_totals)))
ax.set_yticks(np.arange(len(labels_with_totals)))
ax.set_xticklabels(labels_with_totals, rotation=45, ha="right")
ax.set_yticklabels(labels_with_totals)

# Add text in each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cell_text[i][j],
                ha="center", va="center", color="black", fontsize=10)

# Axis labels
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()

# Save
plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix_test_set.png"))
plt.show()
plt.close()
print("Confusion matrix with detailed cell labels saved.")



# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig(os.path.join(RESULTS_PATH, "roc_curve_test.png"))
plt.close()

print("ROC curve saved successfully!")

print("\nEvaluation complete!")
