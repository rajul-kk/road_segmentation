import kagglehub
import os
import shutil

# Set the target directory
target_dir = os.path.join("data", "raw")

# Create the directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("balraj98/deepglobe-road-extraction-dataset")

print("Path to dataset files:", path)

# Copy all files from the downloaded path to the raw folder
if os.path.exists(path):
    print(f"Copying files from {path} to {target_dir}...")
    for item in os.listdir(path):
        source = os.path.join(path, item)
        destination = os.path.join(target_dir, item)
        if os.path.isdir(source):
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    print(f"Dataset copied to {target_dir}")
else:
    print(f"Warning: Downloaded path {path} does not exist")

