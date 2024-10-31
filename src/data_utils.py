import os
import glob
from collections import defaultdict

from typing import List, Tuple
from .config import ORIGINAL_DIR, LABEL_DIR 

def get_file_paths(root: str, file_extension: str) -> List[str]:
    """
    Recursively find all the file paths with the specified file extension.

    Parameters
    ----------
    root : str
        The root directory to conduct the search in.
    file_extension : str
        The file extension to search for (e.g., '.jpg', '.png', '.txt').

    Returns
    -------
    list
        A list of strings that are the file paths of the specified file type.
    """
    # Ensure the file extension starts with a dot
    if not file_extension.startswith('.'):
        file_extension = f'.{file_extension}'

    search_pattern = os.path.join(root, '**', f'*{file_extension}')
    return glob.glob(search_pattern, recursive=True)

def get_valid_data(image_paths: List[str], label_paths: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Prepares valid image-label pairs suitable for TensorFlow model training.
    Each valid pair includes an image and two associated label files that meet criteria:
    one with '_res.tif' and one with '_sunshad.tif' in the filename.

    Parameters
    ----------
    image_paths : List[str]
        List of image file paths.
    label_paths : List[str]
        List of label file paths.

    Returns
    -------
    List[Tuple[str, List[str]]]
        A list of tuples, where each tuple consists of:
        - An image path (str)
        - A list of two label paths (List[str]) corresponding to the image
    """
    # 
    def extract_image_number(file_path: str) -> str:
        """
        Helper function to extract image number from a file path
        with format: .../IMG_0766.jpg
        """
        res = os.path.basename(file_path).split('_')[1].split('.')[0]
        return res

    # Group label paths by image number
    label_dict = defaultdict(list)
    for label in label_paths:
        img_num = extract_image_number(label)
        label_dict[img_num].append(label)

    # Create list of (image_path, label_paths) pairs
    valid_data = []
    for img_path in image_paths:
        img_num = extract_image_number(img_path)
        
        # Filter labels for this image based on criteria
        labels = label_dict.get(img_num, [])
        if len(labels) == 2 and any('_res.tif' in label for label in labels) and any('_sunshad.tif' in label for label in labels):
            valid_data.append((img_path, labels))

    return valid_data

jpg_paths = get_file_paths(ORIGINAL_DIR, 'jpg')
tif_paths = get_file_paths(LABEL_DIR, 'tif')
valid_data = get_valid_data(image_paths=jpg_paths, label_paths=tif_paths)
print(f"Found {len(valid_data)} valid data points")
