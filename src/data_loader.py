import os                      # For interacting with the file system
import glob                    # For file pattern matching
from typing import Tuple, List  # For type hinting
from src.logger import logging  # Custom logging module
from src.exception import CustomException  # Custom exception handling
import sys                     # Needed for exception traceback info

def get_image_paths_and_labels(base_dir: str) -> Tuple[List[str], List[str]]:
    """
    Loads image file paths and their corresponding labels from a given directory.

    Args:
        base_dir (str): The base directory containing class-wise subdirectories.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing a list of image file paths and a list of labels.
    """
    try:
        # List all subdirectories (each representing a class) inside the base directory
        class_dirs = [
            os.path.join(base_dir, cls)
            for cls in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, cls))
        ]

        # Initialize lists to store image paths and their corresponding labels
        image_paths = []
        labels = []

        # Loop over each class directory
        for class_dir in class_dirs:
            # Extract class name from directory name (e.g., 'glioma', 'meningioma', etc.)
            class_name = os.path.basename(class_dir)

            # Gather all .jpg and .png image files in the class directory
            images = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                     glob.glob(os.path.join(class_dir, "*.png"))

            # Add image paths to list
            image_paths.extend(images)

            # Add corresponding labels (class names repeated for number of images)
            labels.extend([class_name] * len(images))

        # Log the total number of images loaded
        logging.info(f"Loaded {len(image_paths)} images from {base_dir}")

        # Return the lists of image paths and labels
        return image_paths, labels

    except Exception as e:
        # Log error if something goes wrong
        logging.error("Failed to load image paths and labels.")
        
        # Raise a custom exception with system information for debugging
        raise CustomException(e, sys)
