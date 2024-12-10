from torch.utils.data import DataLoader
import os
import lightning as L
from ..datasets.default_dataset import DefaultDataset
from ..data_fetchers.RoboflowClient import RoboflowClient
from .helpers import handle_robflow_dataset, handle_google_drive_files
from pathlib import Path
from ..data_fetchers.GoogleDriveClient import GoogleDriveClient
from sklearn.model_selection import train_test_split

class DefaultDatamodule(L.LightningDataModule):
    def __init__(self, data_dir="datasets", batch_size=2, num_workers=4, version = "2", overwrite=False):
        super().__init__()
        print('Init datamodule')
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.version = version
        self.overwrite = overwrite
        self.dataset_location = Path(os.path.join(Path.cwd(), self.data_dir))
        self.google_drive_path = self.dataset_location / "pan_geodeta"
        self.location = self.dataset_location / (os.getenv("ROBOFLOW_PROJECT_NAME") + " --" + self.version)
        self.test_paths = []
        self.test_paths_labels = []

    def prepare_data(self):
        print('Preparing data')
        if not self.google_drive_path.exists():
            google_drive = GoogleDriveClient()
            google_drive.getDataset()
        if not os.path.exists(self.location):
            roboflow = RoboflowClient()
            dataset = roboflow.getDataset(self.version)
            dataset_path = Path(dataset.location)
            folders = [dataset_path / Path(item.name) for item in dataset_path.iterdir() if item.is_dir()]
            handle_robflow_dataset(folders)
            handle_google_drive_files(self.location)


    def setup(self, stage=None):
        # Ścieżki do folderów z danymi
        train_image_dir = self.location / "train/images"
        train_label_dir = self.location / "train/labels"

        all_images = sorted(list(train_image_dir.glob("*.png")))
        all_labels = sorted(list(train_label_dir.glob("*.png")))

        images_train, images_val, labels_train, labels_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
        images_val, images_test, labels_val, labels_test = train_test_split(images_val, labels_val, test_size=0.3, random_state=42)

        self.test_paths_images = images_test
        self.test_paths_labels = labels_test

        # Datasety
        self.train_dataset = DefaultDataset(images_train, labels_train)
        self.val_dataset = DefaultDataset(images_val, labels_val)
        self.test_dataset = DefaultDataset(images_test, labels_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def get_test_paths(self):
        return self.test_paths_images, self.test_paths_labels