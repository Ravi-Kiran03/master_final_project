
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results')

# Load balanced training data
X_train = pd.read_csv(os.path.join(RESULTS_PATH, 'X_train_balanced.csv'))
y_train = pd.read_csv(os.path.join(RESULTS_PATH, 'y_train_balanced.csv')).values.ravel()


# Initialize and train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Save trained model
model_path = os.path.join(RESULTS_PATH, 'random_forest_rq1.pkl')
joblib.dump(rf_model, model_path)

print(f"Random Forest model trained and saved to {model_path}")
