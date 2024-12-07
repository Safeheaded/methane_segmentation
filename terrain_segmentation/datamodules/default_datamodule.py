from torch.utils.data import DataLoader
import os
import lightning as L
from ..datasets.default_dataset import DefaultDataset
from ..roboflow.RoboflowClient import RoboflowClient
from .helpers import handle_robflow_dataset
from pathlib import Path

class DefaultDatamodule(L.LightningDataModule):
    def __init__(self, data_dir="datasets", batch_size=2, num_workers=4, version = "1", overwrite=False):
        super().__init__()
        print('Init datamodule')
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.version = version
        self.overwrite = overwrite
        self.location = Path(os.path.join(Path.cwd(), self.data_dir, os.getenv("ROBOFLOW_PROJECT_NAME") + " --" + self.version))

    def prepare_data(self):
        print('Preparing data')
        if not os.path.exists(self.location):
            roboflow = RoboflowClient()
            dataset = roboflow.getDataset(self.version)
            dataset_path = Path(dataset.location)
            folders = [dataset_path / Path(item.name) for item in dataset_path.iterdir() if item.is_dir()]
            handle_robflow_dataset(folders)


    def setup(self, stage=None):
        # Ścieżki do folderów z danymi
        train_image_dir = self.location / "train/images"
        train_label_dir = self.location / "train/labels"
        val_image_dir = self.location / "valid/images"
        val_label_dir = self.location / "valid/labels"
        test_image_dir = self.location / "test/images"
        test_label_dir = self.location / "test/labels"

        # Datasety
        self.train_dataset = DefaultDataset(train_image_dir, train_label_dir)
        self.val_dataset = DefaultDataset(val_image_dir, val_label_dir)
        self.test_dataset = DefaultDataset(test_image_dir, test_label_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)