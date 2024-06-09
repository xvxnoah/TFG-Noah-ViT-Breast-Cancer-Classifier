from medigan import Generators
import os
from PIL import Image
import numpy as np
import cv2
import shutil


def oversample_training_set(train_data):
    # Load the generated images
    output_path = "generated_images"
    generated_images = os.listdir(output_path)

    # Filter out the benign images
    benign_images = [img for img in generated_images if 'benign' in img.lower()]

    # Preprocess the images
    for img_name in benign_images:
        img_path = os.path.join(output_path, img_name)
        img = Image.open(img_path)
        
        # Convert PIL image to numpy array for processing
        img_array = np.array(img, dtype=np.uint8)

        # Apply median filter for noise reduction
        img_filtered = cv2.medianBlur(img_array, 5)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_filtered)

        # Resize image
        img_size = (224, 224)  # Replace with your desired size
        img_resized = cv2.resize(img_clahe, img_size)

        # Normalize pixel values
        image_np = img_resized / 255.0

        # Check if the image is not completely black
        if np.max(image_np) > 0:
            # Append the processed image and label to train_data
            train_data.append((image_np, 'Benign'))

    # Remove the generated images folder
    shutil.rmtree(output_path)


    return train_data


def generate_images(train_data, images_to_generate):
    # Initialize the Generators object from medigan
    generators = Generators()

    print(f"Generating {images_to_generate} images for oversampling...")

    generators.generate(model_id=8, condition=1, num_samples=images_to_generate, install_dependencies=True, output_path="generated_images")

    # Oversample the training set with the generated images
    train_data = oversample_training_set(train_data)

    return train_data

    