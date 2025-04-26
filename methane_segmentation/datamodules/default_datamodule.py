from torch.utils.data import DataLoader
import os
import lightning as L
from ..datasets.default_dataset import DefaultDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
import albumentations.pytorch.transforms
import pandas as pd

class DefaultDatamodule(L.LightningDataModule):
    def __init__(self, data_dir="datasets", batch_size=4, num_workers=8, exclude_from_input=[]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_location = Path(os.path.join(Path.cwd(), self.data_dir))
        self.test_paths = []
        self.test_paths_labels = []
        self.exclude_from_input = exclude_from_input
        self.inputs = []

        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.pytorch.transforms.ToTensorV2(),
        ])

        self.transforms = A.Compose([
            A.pytorch.transforms.ToTensorV2(),
        ])

    def load_data_dirs(self, path) -> pd.DataFrame:
        return [f for f in Path(path).iterdir() if f.is_dir()]
    
    def get_dirs(self, path):
        possible_labels = ['labelbinary.tif', 'label_rgba.tif']

        excluded_inputs = possible_labels + self.exclude_from_input
        
        inputs = [f for f in path.iterdir() if f.is_file() and f.name not in excluded_inputs]
        self.inputs = [p.stem for p in inputs]
        self.inputs.sort()

    def get_train_dataset_num(self) -> int:
        return int(len(self.get_dirs()) * 0.8)


    def setup(self, stage=None):
        data_dirs = self.load_data_dirs(self.dataset_location)

        self.get_dirs(data_dirs[0])

        X_train_dir_paths, X_test_dir_paths, _, _ = train_test_split(data_dirs, data_dirs, test_size=0.2, random_state=42, shuffle=True)
        X_val_dir_paths, X_test_dir_paths, _, _ = train_test_split(data_dirs, data_dirs, test_size=0.3, random_state=42)

        # Datasety
        self.train_dataset = DefaultDataset(X_train_dir_paths, self.exclude_from_input,transform=self.augmentations)
        self.val_dataset = DefaultDataset(X_val_dir_paths, self.exclude_from_input, transform=self.transforms)
        self.test_dataset = DefaultDataset(X_test_dir_paths, self.exclude_from_input, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
    
    def get_test_paths(self):
        return self.test_paths_images, self.test_paths_labels