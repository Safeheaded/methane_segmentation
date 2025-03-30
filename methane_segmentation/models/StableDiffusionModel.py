import torch
import torch.nn.functional as F
from torch.optim import AdamW

import lightning.pytorch as pl
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline

class StableDiffusionModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-5):
        super().__init__()
        # self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to('mps')
        self.unet = UNet2DModel(
            sample_size=128,
            in_channels=19,
            out_channels=19,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to('mps')
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
        self.learning_rate = learning_rate

        self.pred_logged_epochs = set()

        # self.vae.encoder.conv_in = torch.nn.Conv2d(18, 128, kernel_size=3, stride=1, padding=1)
        # self.vae.decoder.conv_out = torch.nn.Conv2d(128, 18, kernel_size=3, stride=1, padding=1)

    def forward(self, images, timesteps):
        noise_pred = self.unet(images, timesteps).sample
        return noise_pred

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val_loss")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test_loss")

    def shared_step(self, batch, loss_name):
        images, _ = batch

        noise = torch.randn(images.shape, dtype=torch.float32, device=images.device)
        bsz = images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device
        ).long()
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)

        model_output = self(noisy_images, timesteps)

        loss = F.mse_loss(model_output.float(), noise.float())
        self.log(loss_name, loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 1 == 0 and self.current_epoch not in self.pred_logged_epochs:
            self.pred_logged_epochs.add(self.current_epoch)

            with torch.no_grad():
                pipeline = DDPMPipeline(
                    unet=self.unet,
                    scheduler=self.noise_scheduler,
                )
                generator = torch.Generator(device='cpu').manual_seed(0)
                images = pipeline(
                    generator=generator,
                    batch_size=1,
                    num_inference_steps=1000,
                    output_type="np",
                ).images

                # fig, axes = plt.subplots(figsize=(6, 6))

                # Iterujemy po wszystkich 18 kanałach
                # for i in range(19):
                    # Wyciągamy i-ty kanał: shape (512, 512)

                images = (images * 255).round().astype("uint8")

                first_image = images[0]

                # Przygotuj grid – np. 3 wiersze x 6 kolumn (czyli 18 subplots)
                fig, axes = plt.subplots(4, 5, figsize=(15, 8))
                axes = axes.flatten()

                # Iterujemy po wszystkich 18 kanałach
                for i in range(19):
                    # Wyciągamy i-ty kanał: shape (512, 512)
                    channel = first_image[..., i]
                    axes[i].imshow(channel, cmap='gray')
                    axes[i].set_title(f'Channel {i}')
                    axes[i].axis('off')

                plt.tight_layout()
                self.logger.experiment[f"valid/prediction"].append(fig)
                fig.clear()
                plt.close(fig)


        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        # return AdamW(self.unet.parameters(), lr=self.learning_rate)
        return AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
