from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
import rasterio.windows
import pandas as pd
from pathlib import Path
import tifffile

class DiffusionDataset(Dataset):
    def __init__(self, dir_paths: list[Path], exclude_from_inputs: list[str], transform=None):
        self.dir_paths = dir_paths
        self.transform = transform
        self.exclude_from_inputs = exclude_from_inputs
        # preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

    def __len__(self):
        return len(self.dir_paths)
    
    def __getitem__(self, idx):
        # Ścieżki do obrazu i etykiety
        path = self.dir_paths[idx]
        label_path = path / 'labelbinary.tif'
        binary_label_raw = tifffile.imread(label_path)
        label = np.array(binary_label_raw, dtype=np.float32)

        possible_labels = ['label_rgba.tif']

        excluded_inputs = possible_labels + self.exclude_from_inputs
        
        inputs_paths = [f for f in path.iterdir() if f.is_file() and f.name not in excluded_inputs]
        inputs_paths.sort()
        inputs = np.array([np.array(tifffile.imread(s)) / np.max(np.array(tifffile.imread(s))) for s in inputs_paths if Path(s).exists()], dtype=np.float32)    
        return inputs, label



        

        