import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import ssl
from torch.utils.data import DataLoader
from captum.attr import GradientShap, IntegratedGradients
from captum.attr import visualization as viz
from torchmetrics.classification import BinaryAccuracy
from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
from terrain_segmentation.models.default_model import DefaultSegmentationModel
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

    def wrapper(input):
        res = model(input)
        # res = res.squeeze(1)
        return res


    # Wczytaj DataModule
    data_module = DefaultDatamodule(batch_size=8)  # Define your data module
    data_module.setup(stage='test')  # Setup the data for the test stage

    # Pobierz pierwsze 8 próbek z testowego zbioru danych
    test_loader = data_module.test_dataloader()
    inputs = data_module.inputs
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images[3].float()
    labels = labels[3].float()

    label_for_coords = labels.clone().numpy()
    # Sprawdzenie, ile pikseli w macierzy label_for_coords jest równa 1
    num_pixels_equal_to_one = np.sum(label_for_coords == 1)

    print(f"Liczba pikseli równa 1: {num_pixels_equal_to_one}")

    coordinates = np.column_stack(np.where(label_for_coords != 0))

    # Przenieś dane na urządzenie, jeśli jest używane GPU
    device = torch.device('mps')
    images = images.to(device)
    labels = labels.to(device)
    # labels = labels.unsqueeze(0)
    images = images.unsqueeze(0)

    gradient_shap = GradientShap(wrapper)

    # Tworzymy baseline (np. obrazy z zerowymi wartościami)
    baseline = torch.zeros_like(images)
    batch_size, _, height, width = images.shape
    # targets = torch.zeros((1), dtype=torch.long).to(images.device)
    targets = [(0,)] * batch_size  # Każdy obraz ma `target=(0,)`
    all_pixel_coords = list(product(range(width), range(height)))

    # Obliczamy wpływ poszczególnych wejść

    df = pd.DataFrame(columns=['coords', 'layer_index', 'importance'])
    for i, coord in enumerate(coordinates):
        row, col = coord
        print(f"Processing pixel {i}/{num_pixels_equal_to_one}")
        attributions = gradient_shap.attribute(images, baselines=baseline, n_samples=2, target=(0, col, row))
        importance_per_channel = attributions.abs().mean(dim=(0, 2, 3)).cpu()
        topk_values, topk_indices = torch.topk(importance_per_channel, 8)
        for i in range(len(topk_values)):
            importance = topk_values[i].item()
            layer_index = topk_indices[i].item()
            df.loc[len(df)] = {'coords': (row, col), 'layer_index': layer_index, 'importance': importance}

    layer_index_counts = df['layer_index'].value_counts().index[:8]

    print("Najczęściej pojawiające się wartości w kolumnie layer_index:")
    result = [inputs[i] for i in layer_index_counts]
    print(result)

    # attributions = gradient_shap.attribute(images, baselines=baseline, n_samples=2, target=(0,))
    # importance_per_channel = attributions.abs().mean(dim=(0, 2, 3))

    # topk_values, topk_indices = torch.topk(importance_per_channel, 8)

    # print("Indeksy 8 elementów o największych wartościach:", topk_indices)
    # print("Wartości 8 elementów o największych wartościach:", topk_values)

if __name__ == "__main__":
    main()
