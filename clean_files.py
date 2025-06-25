import os
import shutil

dirs_to_clear = [
    './data',
    './output',
]

for file in dirs_to_clear:
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)
        print(f"Cleared file: {file}")
    else:
        print(f"File not found, skipping: {file}")

print("All specified directories and files have been cleared.")

dirs_to_clear.append('./input')

for file in dirs_to_clear:
    os.makedirs(file, exist_ok=True)
    print(f"Created directory: {file}")
    
print("Directories have been created successfully.")