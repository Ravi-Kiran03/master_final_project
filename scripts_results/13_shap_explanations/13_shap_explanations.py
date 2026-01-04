# 13_shap_explanations_global_only.py
# Global SHAP explanations for Random Forest model (RQ2)
# Outputs:
#  - shap_summary_plot.png (global)
#  - shap_bar_plot.png (global)
#  - shap_global_importance.csv (ranked mean(|SHAP|))
#  - shap_global_summary.txt (top-N ranked list)

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Script-specific results folder
# -----------------------------
SCRIPT_RESULTS_NAME = "13_shap_explanations"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "scripts_results")
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)
os.makedirs(RESULTS_PATH, exist_ok=True)

# -----------------------------
# Inputs from previous scripts
# -----------------------------
MODEL_PATH = os.path.join(RESULTS_ROOT, "04_train_random_forest", "random_forest_rq1.pkl")
X_TEST_PATH = os.path.join(RESULTS_ROOT, "02_data_split", "X_test.csv")
Y_TEST_PATH = os.path.join(RESULTS_ROOT, "02_data_split", "y_test.csv")  # optional, not used here

# -----------------------------
# Helpers
# -----------------------------
def get_class1_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """
    Returns SHAP values for the positive class (class index 1) in a robust way.
    Handles both:
      - New SHAP API: explainer(X) -> Explanation with .values shape:
            (n_samples, n_features)  [binary regression-style output]
         or (n_samples, n_features, n_outputs/classes)
      - Older behavior: explainer.shap_values(X) -> list/array
    """
    try:
        sv = explainer(X)  # New API preferred
        values = sv.values

        # If (n, m, 2) -> take class 1
        if values.ndim == 3:
            if values.shape[2] < 2:
                raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
            return values[:, :, 1]

        # If (n, m) -> treat as already for the model output (often positive class / log-odds)
        if values.ndim == 2:
            return values

        raise ValueError(f"Unexpected SHAP values ndim: {values.ndim}, shape: {values.shape}")

    except Exception:
        # Fallback for older SHAP versions
        values = explainer.shap_values(X)

        # For classifiers, often a list of arrays [class0, class1]
        if isinstance(values, list) and len(values) >= 2:
            return np.asarray(values[1])

        # Or sometimes just an array (n, m) for binary output
        values = np.asarray(values)
        if values.ndim == 2:
            return values
        if values.ndim == 3 and values.shape[2] >= 2:
            return values[:, :, 1]

        raise ValueError(f"Could not interpret SHAP output; got type={type(values)} shape={getattr(values,'shape',None)}")


# -----------------------------
# Load model and test data
# -----------------------------
print("Loading model and test data...")
rf_model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
print("Model and data loaded successfully.")

# -----------------------------
# Initialize SHAP explainer
# -----------------------------
print("Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(rf_model)

print("Computing SHAP values (positive / defective class)...")
shap_values_class1 = get_class1_shap_values(explainer, X_test)
print(f"SHAP values computed. Shape: {shap_values_class1.shape}")

# -----------------------------
# Global importance numbers
# -----------------------------
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)  # (n_features,)
importance_df = pd.DataFrame(
    {"feature": X_test.columns, "mean_abs_shap": mean_abs_shap}
).sort_values("mean_abs_shap", ascending=False)

# Save CSV
importance_csv_path = os.path.join(RESULTS_PATH, "shap_global_importance.csv")
importance_df.to_csv(importance_csv_path, index=False)

# Save TXT summary (top N)
top_n = 10
summary_txt_path = os.path.join(RESULTS_PATH, "shap_global_summary.txt")
with open(summary_txt_path, "w") as f:
    f.write("SHAP Global Feature Importance (Random Forest)\n")
    f.write("Positive class: Defective (class 1)\n")
    f.write("=============================================\n\n")
    f.write(f"Top {top_n} features by mean(|SHAP|):\n\n")
    for rank, (_, row) in enumerate(importance_df.head(top_n).iterrows(), start=1):
        f.write(f"{rank:02d}. {row['feature']}: {row['mean_abs_shap']:.6f}\n")

print(f"Saved global SHAP importance CSV to: {importance_csv_path}")
print(f"Saved global SHAP importance TXT to: {summary_txt_path}")

# -----------------------------
# SHAP Summary Plot (Global)
# -----------------------------
print("Generating SHAP summary plot (global)...")
plt.figure()
shap.summary_plot(
    shap_values_class1,
    X_test,
    show=False
)
plt.title("SHAP Summary Plot (Defective Class)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "shap_summary_plot.png"), dpi=300)
plt.close()
print("Saved shap_summary_plot.png")

# -----------------------------
# SHAP Bar Plot (Global Importance)
# -----------------------------
print("Generating SHAP bar plot (global importance)...")
plt.figure()
shap.summary_plot(
    shap_values_class1,
    X_test,
    plot_type="bar",
    show=False
)
plt.title("Global Feature Importance Using SHAP (Defective Class)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "shap_bar_plot.png"), dpi=300)
plt.close()
print("Saved shap_bar_plot.png")

print("\nSHAP global explainability outputs generated successfully!")
