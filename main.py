from dotenv import load_dotenv
from lightning.pytorch import Trainer
from terrain_segmentation.roboflow import RoboflowClient
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from terrain_segmentation.models.default_model import YOLOv5SegmentationModel
import ssl
import torch
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

def main():
    data_module = DefaultDatamodule()

    EPOCHS = 5
    T_MAX = EPOCHS * 15

    model = YOLOv5SegmentationModel(num_classes=1, T_MAX=T_MAX)

    trainer = Trainer(max_epochs=EPOCHS, accelerator='gpu')
    trainer.fit(model=model, datamodule=data_module)

    model_path = "yolov5_segmentation_model3.pth"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()