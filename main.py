from dotenv import load_dotenv
from lightning.pytorch import Trainer
from terrain_segmentation.roboflow import RoboflowClient
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from terrain_segmentation.models.default_model import YOLOv5SegmentationModel
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

def main():
    data_module = DefaultDatamodule()
    model = YOLOv5SegmentationModel(num_classes=1)

    trainer = Trainer(max_epochs=10)
    trainer.fit(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()