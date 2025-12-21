
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Script-specific results folder name
SCRIPT_RESULTS_NAME = "04_train_random_forest"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Common results root folder
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Script-specific results folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Load balanced training data from previous script
X_train = pd.read_csv(os.path.join(RESULTS_ROOT, '03_apply_smote', 'X_train_balanced.csv'))
y_train = pd.read_csv(os.path.join(RESULTS_ROOT, '03_apply_smote', 'y_train_balanced.csv')).values.ravel()


# Initialize and train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, # number of trees
    max_depth=None, # no maximum depth
    random_state=42, # for reproducibility
    n_jobs=-1 # utilize all available cores
)
rf_model.fit(X_train, y_train)

# Save trained model
model_path = os.path.join(RESULTS_PATH, 'random_forest_rq1.pkl')
joblib.dump(rf_model, model_path)

print(f"Random Forest model trained and saved to {model_path}")
