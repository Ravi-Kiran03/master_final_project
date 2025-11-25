# Data Exploration Script for RQ1 that prints dataset overview and saves simple visualizations
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Setting Project root = one level up from scripts
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Data file path
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'kc1_raw.csv')

# Results path
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'RQ1_results')
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


#create plot figure for class distribution
plt.figure(figsize=(6,4)) #setting width and height of the plot
sns.countplot(x='defects', data=df) #countplot to show class distribution setting defects column on x-axis
plt.title("Defective vs Non-Defective Modules") #title of the plot
plt.tight_layout()  #adjust layout to prevent clipping of labels
plot_path = os.path.join(RESULTS_PATH, "01_class_distribution.png") #path to save the plot
plt.savefig(plot_path) #save the plot
plt.close() #close the plot to free memory

print(f"Class distribution plot saved to {plot_path}")
