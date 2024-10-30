import os

# Directory for local data
DATA_DIR = os.getenv("DATA_DIR", "data/")

# Expected directories for images and labels
ORIGINAL_DIR = os.path.join(DATA_DIR, "images/original")
LABEL_DIR = os.path.join(DATA_DIR, "images/label")
