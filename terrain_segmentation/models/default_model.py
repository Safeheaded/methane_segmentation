import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from ultralytics import YOLO
import inference
import torchmetrics
import segmentation_models_pytorch as smp

class YOLOv5SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, input_channels=3, learning_rate=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        
        # Ładujemy pretrenowany model YOLOv5 z detekcją
        self.network = smp.Unet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        # print(self.network)

        self.loss_function = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.save_hyperparameters()
        # print(self.model)
        
        # Modyfikacja modelu do segmentacji
        # Zakładając, że chcemy dodać głowę segmentacji na końcu
        # self.model.segmentation_head = nn.Conv2d(self.model.fpn_out_channels[-1], self.num_classes, kernel_size=1)
        
    def forward(self, x):
        x = x.float()
        
        pred = self.network(x)
        return pred

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        outputs = outputs.squeeze(1)

        loss = self.loss_function(outputs, labels.float())

        self.accuracy.update(outputs, labels.float())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        outputs = outputs.squeeze(1)
        loss = self.loss_function(outputs, labels.float())

        self.accuracy.update(outputs, labels.float())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer