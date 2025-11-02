# Script for initial exploration and preprocessing of the NASA KC1 dataset
# This script provides a quick check of missing values, separates features and labels and generates basic visualizations to understand data distributions and feature correlations.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/Users/ravikiran/Documents/master_final_project/data/kc1_raw.csv' #dataset path
kc1 = pd.read_csv(file_path)

print("Dataset shape:", kc1.shape) #Loading the dataset and displaying basic info
print("First 5 rows:\n", kc1.head())
print("\nDataset info:")
kc1.info()
print("\nMissing values per column:") #Checking missing values
print(kc1.isnull().sum())


X = kc1.drop('defects', axis=1) #Separating features and target
y = kc1['defects']

print("\nFeature summary stats:")
print(X.describe()) #Quick summary statistics

# Graph Visualizations
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Defect vs Non-defect modules")
plt.xlabel("Defect (True/False)")
plt.ylabel("Count")
plt.show()

# Heatmap to see correlations between features
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Check for columns with little or no variation
low_var_cols = [col for col in X.columns if X[col].nunique() <= 1]
print("\nColumns with zero variance (might drop these later):", low_var_cols)

