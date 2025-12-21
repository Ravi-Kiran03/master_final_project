import os
import subprocess
import sys
import re


# Paths

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')


# Get numbered folders
folder_pattern = re.compile(r'^(\d+)_.*$')
folders = []

for name in os.listdir(SCRIPTS_ROOT):
    folder_path = os.path.join(SCRIPTS_ROOT, name)
    if os.path.isdir(folder_path):
        match = folder_pattern.match(name)
        if match:
            folders.append((int(match.group(1)), folder_path))

# Sort folders numerically
folders.sort(key=lambda x: x[0])

if not folders:
    print("No numbered script folders found!")
    sys.exit(1)

print("Folders to process in order:")
for num, folder in folders:
    print(f"{num:02d} -> {folder}")

# Run scripts in each folder

for num, folder in folders:
    # Find scripts starting with the folder number
    script_pattern = re.compile(rf'^{num:02d}_.*\.py$')
    scripts = [f for f in os.listdir(folder) if script_pattern.match(f)]
    scripts.sort()  # just in case multiple scripts in one folder
    
    if not scripts:
        print(f"No scripts found in folder: {folder}")
        continue

    for script in scripts:
        script_path = os.path.join(folder, script)
        print(f"\nRunning script: {script_path}")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error running {script_path}:\n{result.stderr}")
            sys.exit(1)

print("\nAll scripts executed successfully!")
