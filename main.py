from dotenv import load_dotenv
from lightning.pytorch import Trainer
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from terrain_segmentation.models.default_model import DefaultSegmentationModel
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
# that i required to overcome ssl certificate error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# that import is required to stop displaying neptune double value errors
import terrain_segmentation.logging.logging
from datetime import datetime

load_dotenv()

def main():

    neptune_project_name = os.getenv('NEPTUNE_PROJECT_NAME')
    neptune_api_key = os.getenv('NEPTUNE_API_TOKEN')
    EPOCHS = 20
    BATCH_SIZE = 8
    T_MAX = EPOCHS * 344
    learning_rate = 2e-4

    if neptune_project_name and neptune_api_key:
        neptune_logger = NeptuneLogger(
            api_key=neptune_api_key,     # Twój klucz API
            project=neptune_project_name,  # Nazwa projektu w formacie WORKSPACE/PROJECT,
            log_model_checkpoints=False
        )

        PARAMS = {
            "batch_size": BATCH_SIZE,
            "learning_rate": learning_rate,
            "max_epochs": EPOCHS,
            "log_model_checkpoints": False,
            "dependencies": "infer",

        }

        neptune_logger.log_hyperparams(PARAMS)
    current_date = datetime.now().strftime("%d%m%Y%f")
    checkpoint_id = neptune_logger._run_short_id if neptune_logger else current_date
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/epoch/valid/dataset_iou",
        dirpath="trained_models",
        filename=f"{checkpoint_id}_model",
        save_top_k=2,
        mode="max",
        save_last=True,
        every_n_epochs=1,
        save_weights_only=True,
    )


    data_module = DefaultDatamodule(batch_size=BATCH_SIZE)


    model = DefaultSegmentationModel(num_classes=1, T_MAX=T_MAX, learning_rate=learning_rate)

    # trainer = Trainer(max_epochs=EPOCHS, accelerator='gpu', callbacks=[checkpoint_callback])
    trainer = Trainer(max_epochs=EPOCHS, accelerator='gpu', callbacks=[checkpoint_callback], logger=neptune_logger)
    trainer.fit(model=model, datamodule=data_module)

    # neptune_logger.experiment["test/testing_images_paths"].append(", ".join([str(path) for path in data_module.test_paths_images]))
    # neptune_logger.experiment["test/testing_labels_paths"].append(", ".join([str(path) for path in data_module.test_paths_labels]))

    trainer.test(model=model, datamodule=data_module)



if __name__ == "__main__":
    main()