from dotenv import load_dotenv
from lightning.pytorch import Trainer
from terrain_segmentation.roboflow import RoboflowClient
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from terrain_segmentation.models.default_model import DefaultSegmentationModel
import torch
import os
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
# that i required to overcome ssl certificate error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# that import is required to stop displaying neptune double value errors
import terrain_segmentation.logging.logging

load_dotenv()

checkpoint_callback = ModelCheckpoint(
    monitor="metrics/epoch/valid/dataset_iou",
    dirpath="trained_models",
    filename="best_model",
    save_top_k=1,
    mode="max",
)

def main():

    neptune_project_name = os.getenv('NEPTUNE_PROJECT_NAME')
    neptune_api_key = os.getenv('NEPTUNE_API_TOKEN')
    EPOCHS = 1
    T_MAX = EPOCHS * 112

    if neptune_project_name and neptune_api_key:
        neptune_logger = NeptuneLogger(
            api_key=neptune_api_key,     # Twój klucz API
            project=neptune_project_name  # Nazwa projektu w formacie WORKSPACE/PROJECT
        )

        PARAMS = {
            "batch_size": 16,
            "learning_rate": 2e-4,
            "max_epochs": EPOCHS,
        }

        neptune_logger.log_hyperparams(PARAMS)



    data_module = DefaultDatamodule(batch_size=16)


    model = DefaultSegmentationModel(num_classes=1, T_MAX=T_MAX)

    trainer = Trainer(max_epochs=EPOCHS, accelerator='gpu', callbacks=[checkpoint_callback], logger=neptune_logger)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)



if __name__ == "__main__":
    main()