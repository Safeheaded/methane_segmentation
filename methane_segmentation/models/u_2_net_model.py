import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
from .base_model import BaseModel

import math


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode="bilinear", align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(pl.LightningModule):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super().__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate
        )
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(pl.LightningModule):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super().__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f"rebnconv{height}")(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, "downsample")(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f"rebnconv{height}d")(torch.cat((x2, x1), 1))
                return (
                    _upsample_like(x, sizes[height - 1])
                    if not self.dilated and height > 1
                    else x
                )
            else:
                return getattr(self, f"rebnconv{height}")(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module("rebnconvin", REBNCONV(in_ch, out_ch))
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module(f"rebnconv1", REBNCONV(out_ch, mid_ch))
        self.add_module(f"rebnconv1d", REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f"rebnconv{i}", REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(
                f"rebnconv{i}d", REBNCONV(mid_ch * 2, mid_ch, dilate=dilate)
            )

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f"rebnconv{height}", REBNCONV(mid_ch, mid_ch, dilate=dilate))


class U2NET(BaseModel):
    def __init__(self, cfgs, out_ch, learning_rate=2e-4, T_MAX=1000):
        super().__init__(input_channels=18, learning_rate=learning_rate, T_MAX=T_MAX)
        self.out_ch = out_ch
        self._make_layers(cfgs)
        self.loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.loss = nn.BCELoss(size_average=True)
        self.save_hyperparameters()

    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        loss0 = self.loss(d0.squeeze(1), labels_v)
        loss1 = self.loss(d1.squeeze(1), labels_v)
        loss2 = self.loss(d2.squeeze(1), labels_v)
        loss3 = self.loss(d3.squeeze(1), labels_v)
        loss4 = self.loss(d4.squeeze(1), labels_v)
        loss5 = self.loss(d5.squeeze(1), labels_v)
        loss6 = self.loss(d6.squeeze(1), labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

        return loss0, loss

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f"stage{height}")(x)
                x2 = unet(getattr(self, "downsample")(x1), height + 1)
                x = getattr(self, f"stage{height}d")(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f"stage{height}")(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f"side{h}")(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, "outconv")(x)
            maps.insert(0, x)
            return maps

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(
                    f"side{v[0][-1]}", nn.Conv2d(v[2], self.out_ch, 3, padding=1)
                )
        # build fuse layer
        self.add_module(
            "outconv", nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1)
        )

    def shared_step(self, batch, stage="train"):
        inputs, labels = batch
        d0, d1, d2, d3, d4, d5, d6 = self(inputs)
        _, loss = self.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
        self.log("metrics/batch/loss", loss, prog_bar=True)

        outputs = d0[:, 0, :, :]

        outputs =  outputs.sigmoid()
        outputs = (outputs > 0.5).float()

        if stage == "test":
            self.upload_images(outputs, labels)

        tp, fp, fn, tn = smp.metrics.get_stats(
            outputs.long(), labels.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    @classmethod
    def get_U2Net(cls, learning_rate=2e-4, T_MAX=1000):
        full = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            "stage1": ["En_1", (7, 18, 32, 64), -1],
            "stage2": ["En_2", (6, 64, 32, 128), -1],
            "stage3": ["En_3", (5, 128, 64, 256), -1],
            "stage4": ["En_4", (4, 256, 128, 512), -1],
            "stage5": ["En_5", (4, 512, 256, 512, True), -1],
            "stage6": ["En_6", (4, 512, 256, 512, True), 512],
            "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
            "stage4d": ["De_4", (4, 1024, 128, 256), 256],
            "stage3d": ["De_3", (5, 512, 64, 128), 128],
            "stage2d": ["De_2", (6, 256, 32, 64), 64],
            "stage1d": ["De_1", (7, 128, 16, 64), 64],
        }
        return cls(cfgs=full, out_ch=1, learning_rate=learning_rate, T_MAX=T_MAX)
