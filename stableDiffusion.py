from diffusers import StableDiffusionPipeline
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from terrain_segmentation.models.StableDiffusionModel import StableDiffusionModel
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime
from dotenv import load_dotenv
import os
from lightning.pytorch.loggers import NeptuneLogger

load_dotenv()

def main():

    neptune_project_name = os.getenv('SD_NEPTUNE_PROJECT_NAME')
    neptune_api_key = os.getenv('NEPTUNE_API_TOKEN')
    EPOCHS = 1
    BATCH_SIZE = 1
    learning_rate = 1e-5

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


    data_module = DefaultDatamodule(batch_size=BATCH_SIZE)
    model = StableDiffusionModel(learning_rate=learning_rate)
    checkpoint_id = neptune_logger._run_short_id
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="trained_models_stable_diffusion",
        filename=f"{checkpoint_id}_model",
        save_top_k=2,
        mode="max",
        save_last=True,
        every_n_epochs=1,
        save_weights_only=True,
    )

    trainer = Trainer(
        precision="bf16", 
        max_epochs=EPOCHS, 
        accelerator="gpu", callbacks=[checkpoint_callback], 
        logger=neptune_logger
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()