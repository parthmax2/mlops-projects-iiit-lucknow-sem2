import sys
import matplotlib.pyplot as plt
from typing import List
from src.preprocessing import preprocess_image
from src.logger import logging
from src.exception import CustomException


def visualize_preprocessed_images(image_paths: List[str], num_images: int = 5):
    """
    Visualize a few preprocessed images.

    Args:
        image_paths (List[str]): List of image file paths.
        num_images (int): Number of images to display.
    """
    try:
        logging.info(f"Visualizing {num_images} preprocessed images.")
        sample_paths = image_paths[:num_images]
        fig, axs = plt.subplots(1, num_images, figsize=(15, 5))

        for i, path in enumerate(sample_paths):
            img = preprocess_image(path)
            axs[i].imshow(img.squeeze(), cmap='gray')
            axs[i].set_title(f"Image {i+1}")
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error("Error visualizing preprocessed images.")
        raise CustomException(e, sys)
