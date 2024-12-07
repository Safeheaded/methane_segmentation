import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from ultralytics import YOLO
import inference
import torchmetrics

class YOLOv5SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, input_channels=3, learning_rate=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        
        # Ładujemy pretrenowany model YOLOv5 z detekcją
        self.network = YOLO('yolo11n-seg.pt').model
        print(self.network)
        self.save_hyperparameters()

        self.loss_function = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')
        # print(self.model)
        
        # Modyfikacja modelu do segmentacji
        # Zakładając, że chcemy dodać głowę segmentacji na końcu
        # self.model.segmentation_head = nn.Conv2d(self.model.fpn_out_channels[-1], self.num_classes, kernel_size=1)
        
    def forward(self, x):
        x.permute(0, 3, 1, 2)
        # Zwykłe przewidywanie, ale z uwzględnieniem segmentacji
        return self.network(x)['masks']  # Powróć maski obiektów

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)  # przewidywania segmentacji
        loss = self.compute_loss(outputs, masks)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.compute_loss(outputs, masks)
        return loss

    def compute_loss(self, outputs, targets):
        # Definiujemy loss dla segmentacji, np. BCE z logitami
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer