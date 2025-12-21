import os
import pandas as pd
import joblib
from sklearn.svm import SVC


# Paths Setup
SCRIPT_RESULTS_NAME = "09_train_svm"  # give a unique folder name

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Common results root folder
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Script-specific results folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

X_TRAIN_PATH = os.path.join(RESULTS_ROOT, '03_apply_smote', 'X_train_balanced.csv')
Y_TRAIN_PATH = os.path.join(RESULTS_ROOT, '03_apply_smote', 'y_train_balanced.csv')

MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'svm_rq1.pkl')


# Load Train Data
print("Loading training data...")
X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()

print("Training data loaded successfully!")


# Train SVM Model
print("\nTraining SVM model...")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

print("Model training completed!")


# Save Model
joblib.dump(svm_model, MODEL_SAVE_PATH)
print(f"\nSVM model saved successfully at:\n{MODEL_SAVE_PATH}")
