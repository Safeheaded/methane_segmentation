from captum.attr import FeatureAblation
from terrain_segmentation.models.default_model import DefaultSegmentationModel
import torch
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from captum.attr import visualization as viz

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
    inputs = data_module.inputs
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.float()
    labels = labels.float()

    device = torch.device('mps')
    images = images.to(device)
    labels = labels.to(device)
    # images = images.unsqueeze(0)

    out = model(images)

    # Find most likely segmentation class for each pixel.
    out_max = torch.argmax(out, dim=1, keepdim=True)

    def agg_segmentation_wrapper(inp):
        model_out = model(inp)
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))

    # Tworzymy obiekt ablacji
    ablation = FeatureAblation(agg_segmentation_wrapper)

    # Przygotowanie feature_mask:
    # Założenie: images o wymiarach (1, C, H, W)
    B, C, H, W = images.shape
    feature_mask = torch.arange(C, device=images.device).view(1, C, 1, 1).expand(B, C, H, W)

    # Obliczamy wpływ poszczególnych kanałów wejściowych przez ablację całych kanałów
    # przekazując feature_mask, która grupuje piksele w kanale razem
    attributions = ablation.attribute(images, target=0, feature_mask=feature_mask)
    # viz.visualize_image_attr(attributions[0].cpu().detach().permute(1,2,0).numpy(),sign="all")

    # Wynik to tensor z wartościami wpływu każdego kanału (1, C, H, W)
    print(attributions.shape)
    features = data_module.inputs

    # Uśrednienie wpływu po wymiarach przestrzennych (H, W), żeby uzyskać wpływ per channel
    channel_importance = attributions.abs().mean(dim=(2, 3)).sum(dim=(0)).tolist()
    print(channel_importance)
    res = list(zip(features, channel_importance))
    res.sort(key=lambda x: x[1], reverse=True)
    print(res)

if __name__ == '__main__':
    main()
