from dotenv import load_dotenv
from lightning.pytorch import Trainer
from terrain_segmentation.roboflow import RoboflowClient
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from terrain_segmentation.models.default_model import YOLOv5SegmentationModel
import ssl
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

checkpoint_callback = ModelCheckpoint(
    monitor="valid_dataset_iou",
    dirpath="trained_models",
    filename="best_model",
    save_top_k=1,
    mode="max",
)

def main():
    data_module = DefaultDatamodule()

    EPOCHS = 20
    T_MAX = EPOCHS * 15

    model = YOLOv5SegmentationModel(num_classes=1, T_MAX=T_MAX)

    trainer = Trainer(max_epochs=EPOCHS, accelerator='gpu', callbacks=[checkpoint_callback])
    trainer.fit(model=model, datamodule=data_module)

    model_path = "yolov5_segmentation_model3.pth"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()