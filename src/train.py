import os
from config import DATA_DIR

label_dir = os.path.join(DATA_DIR, "images/label")
original_dir = os.path.join(DATA_DIR, "images/original")

def list_files_in_directory(directory):
    """List all files in a given directory."""
    try:
        files = os.listdir(directory)
        print(f"Contents of {directory}:")
        for file in files:
            print(file)
        print()  # Empty line for better readability
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
    except PermissionError:
        print(f"Permission denied for accessing {directory}.")

# List contents of each folder
list_files_in_directory(label_dir)
list_files_in_directory(original_dir)
