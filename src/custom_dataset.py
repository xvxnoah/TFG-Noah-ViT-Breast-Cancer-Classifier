import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from medigan_generator import generate_images
from utils import count_labels_set

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, downsample=False, oversample=False):
        self.label_map = {'Benign': 0, 'Malignant': 1} # Map labels to integers
        filtered_data = [(img, lbl) for img, lbl in data if lbl in self.label_map] # Filter out data with labels not in the label_map

        # Downsample data to balance classes
        if downsample:
            filtered_data = self.downsample(filtered_data)

        # Generate images to balance classes
        if oversample:
            _, label_counts = count_labels_set(filtered_data)
            images_to_generate = (label_counts['Malignant'] - label_counts['Benign'])
            filtered_data = generate_images(filtered_data, images_to_generate)
        
        random.shuffle(filtered_data)  # Shuffle to mix benign and malignant samples

        self.data = filtered_data
        self.transform = transform

    def downsample(self, data):
        benign_data = [item for item in data if item[1] == 'Benign']
        malignant_data = [item for item in data if item[1] == 'Malignant']

        # Find the minimum count between benign and malignant
        min_count = min(len(benign_data), len(malignant_data))
        benign_data = random.sample(benign_data, min_count) # Downsample benign data to min_count
        malignant_data = random.sample(malignant_data, min_count) # Downsample malignant data to min_count

        return benign_data + malignant_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        label = self.label_map[label]

        if isinstance(image, Image.Image):
                image = np.array(image) # Convert PIL image to numpy array if needed

        # Convert grayscale image to RGB by repeating the grayscale channel three times
        if image.ndim == 2:  # Only for 2D grayscale images
            image = np.stack([image]*3, axis=-1)  # Stack to make RGB

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert HxWxC to CxHxW

        # Apply the transformation if available
        if self.transform:
                image = self.transform(image)

        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label