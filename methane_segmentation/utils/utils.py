import os
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from datetime import datetime
from lightning.pytorch import LightningModule, Trainer
from methane_segmentation.datamodules.default_datamodule import DefaultDatamodule
from methane_segmentation.datamodules.diffusion_datamodule import DiffusionDatamodule
from methane_segmentation.models.base_model import BaseModel


def prepare_logger(
    model_name: str, epochs: int, batch_size: int, learning_rate: float
) -> tuple[NeptuneLogger, ModelCheckpoint]:
    neptune_project_name = os.getenv(f"NEPTUNE_NAME_{model_name.upper()}")
    neptune_api_key = os.getenv("NEPTUNE_API_TOKEN")
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    learning_rate = learning_rate

    if neptune_project_name and neptune_api_key:
        neptune_logger = NeptuneLogger(
            api_key=neptune_api_key,  # Twój klucz API
            project=neptune_project_name,  # Nazwa projektu w formacie WORKSPACE/PROJECT,
            log_model_checkpoints=False,
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
        monitor="training/val_loss" if model_name == "diffussion" else "metrics/epoch/valid/dataset_iou",
        dirpath="trained_models",
        filename=f"{checkpoint_id}_model",
        save_top_k=2,
        mode="max",
        save_last=True,
        every_n_epochs=1,
        save_weights_only=True,
    )
    return neptune_logger, checkpoint_callback


def train(
    model: BaseModel,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    accelerator="gpu",
    model_name: str = "unet",
):
    # Prepare the logger and checkpoint callback
    neptune_logger, checkpoint_callback = prepare_logger(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Initialize the data module and model
    data_module = (
        DefaultDatamodule(batch_size=batch_size)
        if model_name != "diffussion"
        else DiffusionDatamodule(batch_size=batch_size)
    )

    T_MAX = data_module.get_train_dataset_num()

    model.set_tmax(T_MAX)

    # Create the trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
        logger=neptune_logger,
        precision="bf16-mixed",
    )

    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Test the model
    trainer.test(model=model, datamodule=data_module)
