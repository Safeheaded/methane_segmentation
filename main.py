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
    model = YOLOv5SegmentationModel(num_classes=1)

    trainer = Trainer(max_epochs=20, accelerator='gpu')
    trainer.fit(model=model, datamodule=data_module)

    data_module.setup(stage='test')
    test_dataset = data_module.test_dataloader().dataset
    first_image, first_mask = test_dataset[0]
    model_path = "yolov5_segmentation_model2.pth"
    torch.save(model.state_dict(), model_path)
    model.eval()
    model.freeze()
    
    with torch.no_grad():
        first_image = first_image.unsqueeze(0)  # Dodaj wymiar batch
        prediction = model(first_image)
        prediction = prediction.squeeze(0).squeeze(0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(first_image.squeeze().permute(1, 2, 0).cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(first_mask.cpu().numpy(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(prediction.cpu().numpy(), cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()