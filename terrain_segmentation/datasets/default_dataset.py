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

class DefaultDataset(Dataset):
    def __init__(self, dir_paths: list[Path], exclude_from_inputs: list[str], with_rgba_labels: bool, transform=None):
        self.dir_paths = dir_paths
        self.transform = transform
        self.exclude_from_inputs = exclude_from_inputs
        self.with_rgba_labels = with_rgba_labels
        # preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

    def __len__(self):
        return len(self.dir_paths)
    
    def __getitem__(self, idx):
        # Ścieżki do obrazu i etykiety
        path = self.dir_paths[idx]
        label_path = path / 'labelbinary.tif'
        binary_label_raw = tifffile.imread(label_path)
        label = np.array(binary_label_raw, dtype=np.float32)

        possible_labels = ['labelbinary.tif', 'label_rgba.tif']

        excluded_inputs = possible_labels + self.exclude_from_inputs

        if self.with_rgba_labels:
            label_path = path / 'label_rgba.tif'
            rgba_label = np.array(tifffile.imread(label_path))
            rgba_label = np.transpose(rgba_label, (2, 0, 1))
            label = np.concatenate((rgba_label, label[None, :, :]), axis=0)
        inputs_paths = [f for f in path.iterdir() if f.is_file() and f.name not in excluded_inputs]
        inputs_paths.sort()
        inputs = np.array([np.array(tifffile.imread(s)) for s in inputs_paths if Path(s).exists()], dtype=np.float32)
        return inputs, label



        

        