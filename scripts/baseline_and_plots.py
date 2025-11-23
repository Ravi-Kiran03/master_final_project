# Trains baseline machine learning models (Decision Tree, Random Forest, SVM),
# Evaluates them using 5-fold stratified cross-validation, and plots the class distribution.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os

os.makedirs("results/figures", exist_ok=True) 

# Load KC1 dataset
kc1 = pd.read_csv("/Users/ravi/Documents/master_final_project/data/kc1_raw.csv")
X = kc1.drop("defects", axis=1)       # Features
y = kc1["defects"].astype(int)        # Target

# Baseline models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": Pipeline([
        ('scaler', StandardScaler()),  # Scale features for SVM
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])
}

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    results.append({"Model": name, "F1 Score": f"{f1.mean():.3f}", "AUC-ROC": f"{auc.mean():.3f}"})

# Print Table
print(pd.DataFrame(results).to_markdown(index=False))

# Plot class distribution
plt.figure(figsize=(6,4.5))
sns.countplot(x=y, hue=y, palette={0: 'lightblue', 1: 'salmon'}, legend=False)
plt.title('KC1 Class Distribution', fontweight='bold')
plt.xlabel('Defects')
plt.ylabel('Count')
plt.xticks([0,1], ['Non-Defective', 'Defective'])
for i, v in enumerate(y.value_counts().sort_index()):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("results/figures/kc1_class_dist.png", dpi=300)
plt.close()
