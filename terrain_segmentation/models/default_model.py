import segmentation_models_pytorch as smp
from .base_model import BaseModel

class DefaultSegmentationModel(BaseModel):

    def __init__(self, num_classes = 1, input_channels=18, learning_rate=1e-3, encoder_name="efficientnet-b0", T_MAX=100, encoder_weights="imagenet", model_name="Unet"):
        super().__init__(input_channels=input_channels, learning_rate=learning_rate, T_MAX=T_MAX)
        self.num_classes = num_classes

        model = getattr(smp, model_name) if hasattr(smp, model_name) else None
        if model is None:
            raise ValueError(f"Model {model_name} not found in segmentation_models_pytorch.")
        
        self.network = model(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=input_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
        
        

        self.loss_function = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.save_hyperparameters()
        
    def forward(self, image):
        image = image.float()
        image = (image - image.mean(dim=(2, 3), keepdim=True)) / (image.std(dim=(2, 3), keepdim=True) + 1e-8)
        res = self.network(image)
        return res
    
    @classmethod
    def get_Unet(cls, T_MAX=100, learning_rate=1e-3, encoder_name="efficientnet-b0", encoder_weights="imagenet"):
        return cls(num_classes=1, input_channels=18, learning_rate=learning_rate, T_MAX=T_MAX, encoder_name=encoder_name, encoder_weights=encoder_weights, model_name="Unet")
    
    @classmethod
    def get_segformer(cls, T_MAX=100, learning_rate=1e-3, encoder_name="efficientnet-b0", encoder_weights="imagenet"):
        return cls(num_classes=1, input_channels=18, learning_rate=learning_rate, T_MAX=T_MAX, encoder_name=encoder_name, encoder_weights=encoder_weights, model_name="Segformer")