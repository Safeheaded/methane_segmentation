from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn

class DefaultDataset(Dataset):
    def __init__(self, images: list[str], labels: list[str], transform=None):
        self.image_dir = images
        self.label_dir = labels
        self.transform = transform
        self.image_paths = images
        self.label_paths = labels
        # preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Ścieżki do obrazu i etykiety
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Załaduj obraz
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image).copy()
        image = np.resize(image, (512, 512, 3)).transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        mask = Image.open(label_path)
        mask = np.asarray(mask).copy()
        mask = torch.tensor(mask, dtype=torch.float32) / 255.0 



        # Zwróć obraz i maski
        return image, mask