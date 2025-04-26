from abc import ABC, abstractmethod
import lightning.pytorch as pl
import torch
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

class BaseModel(ABC, pl.LightningModule):

    training_step_outputs = []
    validation_step_outputs = []
    test_step_outputs = []

    def __init__(self, input_channels=18, learning_rate=1e-3, T_MAX=100):
        super().__init__()
        self.T_MAX = T_MAX
        self.input_channels = input_channels
        self.learning_rate = learning_rate

    @abstractmethod
    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch)

        self.training_step_outputs.append(train_loss_info)

        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch)

        self.validation_step_outputs.append(valid_loss_info)

        return valid_loss_info
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return
    
    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, 'test')
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=720, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def shared_step(self, batch, stage="train"):
        inputs, labels = batch
        outputs = self(inputs)
        outputs = outputs.squeeze(1)
        loss = self.loss_function(outputs, labels)
        self.log("metrics/batch/loss", loss, prog_bar=True)

        # outputs =  outputs.sigmoid()
        outputs = (outputs > 0.5).float()

        if stage == "test":
            self.upload_images(outputs, labels)

        tp, fp, fn, tn = smp.metrics.get_stats(
            outputs.long(), labels.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    def upload_images(self, outputs, labels):
        amount = outputs.shape[0]
        for i in range(amount):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # Ground Truth
            axes[0].imshow(labels.cpu()[i, :, :], cmap='gray')
            axes[0].set_title("Ground Truth")
            axes[0].axis('off')
            # Prediction
            axes[1].imshow(outputs.cpu()[i, :, :], cmap='gray')
            axes[1].set_title("Prediction")
            axes[1].axis('off')
            # plt.tight_layout()
            # Zapis wykresu do bufora
            # Przesłanie obrazu za pomocą loggera
            self.logger.experiment[f"test/prediction"].append(fig)
            fig.clear()
            plt.close(fig)
    
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        self.log(f"metrics/epoch/{stage}/per_image_iou", per_image_iou, prog_bar=True)

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.log(f"metrics/epoch/{stage}/dataset_iou", dataset_iou, prog_bar=True)
        # metrics = {
        #     f"{stage}_per_image_iou": per_image_iou,
        #     f"{stage}_dataset_iou": dataset_iou,
        # }

        # self.log_dict(metrics, prog_bar=True)
