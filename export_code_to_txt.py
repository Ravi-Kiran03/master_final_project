import os
import re

# Root paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_ROOT = os.path.join(PROJECT_ROOT, 'scripts_results')

# Output file
OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    '23037810-Ravi-Kiran-Malishetti MLModels.txt'
)

# Regex for numbered folders (e.g. 01_xxx)
folder_pattern = re.compile(r'^(\d+)_.*$')

folders = []

# Collect numbered folders
for name in os.listdir(SCRIPTS_ROOT):
    folder_path = os.path.join(SCRIPTS_ROOT, name)
    if os.path.isdir(folder_path):
        match = folder_pattern.match(name)
        if match:
            folders.append((int(match.group(1)), folder_path))

# Sort folders numerically
folders.sort(key=lambda x: x[0])

if not folders:
    raise RuntimeError("No numbered script folders found.")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    out.write("MSc Project Artefact Code\n")
    out.write("Student: Ravi Kiran Malishetti (23037810)\n")
    out.write("=" * 80 + "\n\n")

    for _, folder_path in folders:
        scripts = [
            f for f in os.listdir(folder_path)
            if f.endswith('.py')
        ]
        scripts.sort()

        for script in scripts:
            script_path = os.path.join(folder_path, script)

            out.write("\n" + "-" * 80 + "\n")
            out.write(f"SCRIPT: {script}\n")
            out.write("-" * 80 + "\n\n")

            with open(script_path, 'r', encoding='utf-8') as f:
                out.write(f.read())

            out.write("\n\n")

print(f"All scripts exported successfully to:\n{OUTPUT_FILE}")
