from dotenv import load_dotenv
from methane_segmentation.models.default_model import DefaultSegmentationModel
from methane_segmentation.models.u_2_net_model import U2NET
from methane_segmentation.models.acc_unet import AccUnet
import typer
from methane_segmentation.utils.utils import train
from methane_segmentation.models.StableDiffusionModel import StableDiffusionModel

load_dotenv()

app = typer.Typer()


@app.command()
def unet(batch_size: int = 8, epochs: int = 100, learning_rate: float = 2e-4):
    """
    Train a U-Net model.
    """

    model = DefaultSegmentationModel.get_Unet(learning_rate=learning_rate).to("cuda")

    train(model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, model_name="unet")


@app.command()
def acc_unet(batch_size: int = 8, epochs: int = 100, learning_rate: float = 1e-3):
    """
    Train an AccU-Net model.
    """

    model = AccUnet(learning_rate=learning_rate).to("cuda")

    train(model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, model_name="accUnet")

@app.command()
def u2net(batch_size: int = 8, epochs: int = 100, learning_rate: float = 2e-4):
    """
    Train a U2Net model.
    """

    model = U2NET.get_U2Net(learning_rate=learning_rate).to("cuda")

    train(model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, model_name="u2net")

@app.command()
def segformer(batch_size: int = 8, epochs: int = 100, learning_rate: float = 2e-4):
    """
    Train a Segformer model.
    """

    model = DefaultSegmentationModel.get_segformer(learning_rate=learning_rate).to("cuda")

    train(model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, model_name="segformer")

@app.command()
def diffusion(batch_size: int = 8, epochs: int = 100, learning_rate: float = 2e-4, ckpt_path: str = None):
    """
    Train a Diffusion model.
    """

    model = StableDiffusionModel(learning_rate=learning_rate).to("cuda") if ckpt_path is None else StableDiffusionModel.load_from_checkpoint(ckpt_path, learning_rate=learning_rate).to("cuda")

    train(model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, model_name="diffussion")


if __name__ == "__main__":
    app()
