from terrain_segmentation.datamodules.default_datamodule import DefaultDatamodule
import matplotlib.pyplot as plt
from terrain_segmentation.models.StableDiffusionModel import StableDiffusionModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UNet2DModel, DDPMPipeline
import torch

def main():
    data_module = DefaultDatamodule(batch_size=8)
    data_module.setup()
    dataloader = data_module.test_dataloader()
    features = data_module.inputs

    model = StableDiffusionModel.load_from_checkpoint('trained_models_stable_diffusion/14032025024825_model.ckpt')
    model.eval()
    batch = next(iter(dataloader))
    images, labels = batch
    
    # Pierwszy obrazek z batcha
    first_image = images[0]  # shape: [18, 512, 512]
    
    # Ustawiamy grid, np. 3 wiersze x 6 kolumn (czyli 18 subplots)
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.flatten()
    
    # Iterujemy po wszystkich 18 kanałach
    for i in range(18):
        channel = first_image[i].cpu().detach().numpy()  # zamiana tensora na numpy array
        axes[i].imshow(channel, cmap='gray')
        axes[i].set_title(features[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
    pipeline = DDPMPipeline(
        unet=model.unet,
        scheduler=noise_scheduler,
    )
    generator = torch.Generator(device='cpu').manual_seed(0)
    images = pipeline(
        generator=generator,
        batch_size=1,
        num_inference_steps=1000,
        output_type="np",
    ).images

    images = (images * 255).round().astype("uint8")

    print("Shape images: ", images.shape)  # (1, 18, 512, 512)

    first_image = images[0]

    # Przygotuj grid – np. 3 wiersze x 6 kolumn (czyli 18 subplots)
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.flatten()

    # Iterujemy po wszystkich 18 kanałach
    for i in range(18):
        # Wyciągamy i-ty kanał: shape (512, 512)
        channel = first_image[..., i]
        axes[i].imshow(channel, cmap='gray')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
