import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import ssl
from torch.utils.data import DataLoader
from captum.attr import GradientShap, IntegratedGradients
from captum.attr import visualization as viz
from torchmetrics.classification import BinaryAccuracy
from methane_segmentation.datamodules.default_datamodule import DefaultDatamodule
from methane_segmentation.models.default_model import DefaultSegmentationModel
import torch.multiprocessing
from itertools import product
import pandas as pd


def main():
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_default_dtype(torch.float32)
    torch.mps.empty_cache()  # Czyści cache na MPS
    T_MAX = 50
    learning_rate = 2e-4

    # Wczytaj model U-Net
    checkpoint_path = '/Users/patryk/masters/methane_segmentation/trained_models/MET-131_model.ckpt'
    model = DefaultSegmentationModel.load_from_checkpoint(checkpoint_path, num_classes=1, T_MAX=50)
    model = model.to(torch.device('mps'))  # Przenieś model na MPS
    model.eval()  # Set the model to evaluation mode

    # Wczytaj DataModule
    data_module = DefaultDatamodule(batch_size=14)  # Define your data module
    data_module.setup(stage='test')  # Setup the data for the test stage

    # Pobierz pierwsze 8 próbek z testowego zbioru danych
    test_loader = data_module.test_dataloader()
    inputs = data_module.inputs
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.float()
    labels = labels.float()

    label_for_coords = labels.clone().numpy()
    # Sprawdzenie, ile pikseli w macierzy label_for_coords jest równa 1
    num_pixels_equal_to_one = np.sum(label_for_coords == 1)

    coordinates = np.column_stack(np.where(label_for_coords != 0))

    # Przenieś dane na urządzenie, jeśli jest używane GPU
    device = torch.device('mps')
    images = images.to(device)
    labels = labels.to(device)
    # labels = labels.unsqueeze(0)
    # images = images.unsqueeze(0)

    out = model(images)

    # Find most likely segmentation class for each pixel.
    out_max = torch.argmax(out, dim=1, keepdim=True)
    features = data_module.inputs

    def agg_segmentation_wrapper(inp):
        model_out = model(inp)
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))

    gradient_shap = GradientShap(agg_segmentation_wrapper)

    # Tworzymy baseline (np. obrazy z zerowymi wartościami)
    baseline = torch.zeros_like(images)
    batch_size, _, height, width = images.shape
    # targets = torch.zeros((1), dtype=torch.long).to(images.device)
    targets = [(0,)] * batch_size  # Każdy obraz ma `target=(0,)`

    attributions = gradient_shap.attribute(images, baselines=baseline, n_samples=2, target=0)
    channel_importance = attributions.abs().mean(dim=(0, 2, 3)).tolist()
    res = list(zip(features, channel_importance))
    res.sort(key=lambda x: x[1], reverse=True)
    print(res)

if __name__ == "__main__":
    main()
