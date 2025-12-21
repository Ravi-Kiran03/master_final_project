# Data Exploration Script for RQ1 that prints dataset overview and saves simple visualizations
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This script directory
SCRIPT_RESULTS_NAME = "01_data_exploration"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Setting Project root = one level up from scripts
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Data file path
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'kc1_raw.csv')

# Results path
# Results root folder
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Script-specific results folder
RESULTS_PATH = os.path.join(RESULTS_ROOT, SCRIPT_RESULTS_NAME)

# Create folder if it does not exist
os.makedirs(RESULTS_PATH, exist_ok=True)


# Load dataset and print basic info
df = pd.read_csv(DATA_PATH)

print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset shape (rows, columns):", df.shape, "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

print("Class distribution (counts):")
print(df['defects'].value_counts())

# Percentages
print("\nClass distribution (percentages):")
print(df['defects'].value_counts(normalize=True) * 100)

# Create plot figure for class distribution
plt.figure(figsize=(6,4))
ax = sns.countplot(x='defects', data=df)

# Title
plt.title("Defective vs Non-Defective Modules")

# Rename x-axis labels if desired
ax.set_xticklabels(['Non-Defective', 'Defective'])

# Add counts + percentages on top of bars
total = len(df)  # total number of rows
for p in ax.patches:
    height = p.get_height()
    percentage = height / total * 100
    ax.annotate(f'{height} ({percentage:.1f}%)',  # show count + percentage
                (p.get_x() + p.get_width() / 2, height), 
                ha='center', 
                va='bottom') 

plt.tight_layout()  # adjust layout to prevent clipping

# Save plot
plot_path = os.path.join(RESULTS_PATH, "01_class_distribution.png")
plt.savefig(plot_path)
plt.close()

print(f"Class distribution plot saved to {plot_path}")