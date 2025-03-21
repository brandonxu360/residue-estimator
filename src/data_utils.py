import os
import glob
from collections import defaultdict
from typing import List, Tuple
import cv2
import numpy as np
import tensorflow as tf

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

def process_labels(res_label_path: str, sunshad_label_path: str) -> np.ndarray:
    """
    Reads and processes two label images to combine them into a single mask 
    with four classes:
        - 0: Nonresidue & Shaded
        - 1: Nonresidue & Sunlit
        - 2: Residue & Shaded
        - 3: Residue & Sunlit

    Parameters
    ----------
    res_label_path : str
        Path to the residue label file.
    sunshad_label_path : str
        Path to the sun/shade label file.

    Returns
    -------
    np.ndarray
        A combined mask with four unique class labels.
    """
    # Load the labels
    res_label = cv2.imread(res_label_path, cv2.IMREAD_UNCHANGED)
    sunshad_label = cv2.imread(sunshad_label_path, cv2.IMREAD_UNCHANGED)

    # Convert labels to binary
    res_label = np.where(res_label == 255, 1, 0)  # Nonresidue: 0, Residue: 1
    sunshad_label = np.where(sunshad_label == 255, 1, 0)  # Shaded: 0, Sunlit: 1

    # Combine the labels to get four classes
    combined_label = 2 * res_label + sunshad_label
    return combined_label

def class_distribution(masks: np.ndarray) -> dict:
    """
    Calculate the distribution of classes in the given masks.

    Parameters
    ----------
    masks : np.ndarray
        A batch of masks.

    Returns
    -------
    dict
        A dictionary with class labels as keys and their counts as values.
    """
    unique_classes, counts = np.unique(masks, return_counts=True)
    return dict(zip(unique_classes, counts))


# Data Generator for Loading Batches Efficiently
def data_generator(valid_data: List[Tuple[str, List[str]]], batch_size: int):
    while True:
        batch_images = []
        batch_masks = []
        
        for i in range(batch_size):
            print(valid_data[i])
            img_path, mask_paths = valid_data[i]
            
            # Load image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.cast(img, tf.float32) / 255.0  # Convert to float32 and normalize
            
            # Process and resize combined label
            combined_label = process_labels(mask_paths[0], mask_paths[1])
            print("Class Distribution:", class_distribution(combined_label))  # Print class distribution
            
            batch_images.append(img)
            batch_masks.append(combined_label)
        
        yield np.array(batch_images), np.array(batch_masks)

