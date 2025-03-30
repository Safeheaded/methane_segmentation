from captum.attr import FeatureAblation
from methane_segmentation.models.default_model import DefaultSegmentationModel
import torch
from methane_segmentation.datamodules.default_datamodule import DefaultDatamodule

def main():
    T_MAX = 50
    learning_rate = 2e-4

    # Wczytaj model U-Net
    checkpoint_path = '/Users/patryk/masters/methane_segmentation/trained_models/MET-131_model.ckpt'
    model = DefaultSegmentationModel.load_from_checkpoint(checkpoint_path, num_classes=1, T_MAX=50)
    model = model.to(torch.device('mps'))  # Przenieś model na MPS
    model.eval()  # Set the model to evaluation mode

    data_module = DefaultDatamodule(batch_size=32)  # Define your data module
    data_module.setup(stage='test')  # Setup the data for the test stage

    # Pobierz pierwsze 8 próbek z testowego zbioru danych
    test_loader = data_module.test_dataloader()
    features = data_module.inputs
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.float()
    labels = labels.float()

    device = torch.device('mps')
    images = images.to(device)
    labels = labels.to(device)
    # images = images.unsqueeze(0)

    def channel_ablation(model, input_tensor):
        num_channels = input_tensor.shape[1]
        original_output = model(input_tensor).detach()

        impact_scores = []
        for i in range(num_channels):
            modified_input = input_tensor.clone()
            modified_input[:, i, :, :] =  modified_input[:, i, :, :].mean() # Zerowanie kanału
            new_output = model(modified_input).detach()
            impact = (original_output - new_output).abs().mean().item()
            impact_scores.append(impact)

        return impact_scores
    
    impact_scores = channel_ablation(model, images)
    print(impact_scores)

    res = list(zip(features, impact_scores))
    res.sort(key=lambda x: x[1], reverse=True)
    print(res)
    
if __name__ == "__main__":
    main()