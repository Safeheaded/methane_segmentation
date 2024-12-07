import os
from roboflow import Roboflow
import shutil

class RoboflowClient:
    def __init__(self):
        self.api_key = os.getenv("ROBOFLOW_API_KEY")

        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY not set in environment variables")
        
        self.rf = Roboflow(api_key=self.api_key)
        
    def getDataset(self, dataset_id):
        project = self.rf.workspace("automatic-and-robotic").project("segmentacja-terenow")

        dataset = project.version('1').download("coco")

        target_folder = f"datasets/{dataset.name}"

        # Tworzenie folderu, jeśli nie istnieje
        os.makedirs(target_folder, exist_ok=True)

        # Przenoszenie plików
        shutil.move(dataset.location, target_folder)

        return dataset