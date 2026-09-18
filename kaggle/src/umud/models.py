"""U-Net with a geometry-aware orientation head.

The segmentation branch is a conventional U-Net over a ResNet encoder — the
architecture every baseline in this field uses, and deliberately so: the point
of the project is not a better segmenter but a better way of turning a
segmentation into a measurement.

The second branch is the contribution.  Instead of recovering fascicle
orientation afterwards with a structure tensor or a Hough transform, the network
predicts a dense orientation field directly, as a doubled-angle unit vector
``(cos 2t, sin 2t)`` per pixel.  Doubling removes the 180-degree ambiguity that
makes raw angle regression discontinuous along every fascicle, and predicting a
*field* rather than a single angle means the downstream aggregation can weight
by segmentation confidence and report its own dispersion.

The two branches share the encoder, so the orientation task also acts as an
auxiliary signal for the segmentation task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ENCODER_CHANNELS = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
}


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # Odd input sizes make the decoder output drift by a pixel.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    """torchvision ResNet adapted to single-channel ultrasound input.

    B-mode ultrasound is grayscale.  Rather than tile the channel three times we
    sum the pretrained RGB stem weights, which preserves the filter responses
    exactly for a gray input while cutting the stem's parameter count.
    """

    def __init__(self, name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        import torchvision

        weights = "DEFAULT" if pretrained else None
        try:
            net = getattr(torchvision.models, name)(weights=weights)
            self.pretrained = pretrained
        except Exception:
            # No network access in the kernel — fall back rather than crash.
            net = getattr(torchvision.models, name)(weights=None)
            self.pretrained = False

        stem = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            stem.weight.copy_(net.conv1.weight.sum(dim=1, keepdim=True))
        net.conv1 = stem

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.channels = _ENCODER_CHANNELS[name]

    def forward(self, x):
        f0 = self.stem(x)               # 1/2
        f1 = self.layer1(self.pool(f0))  # 1/4
        f2 = self.layer2(f1)             # 1/8
        f3 = self.layer3(f2)             # 1/16
        f4 = self.layer4(f3)             # 1/32
        return [f0, f1, f2, f3, f4]


class GeometryAwareUNet(nn.Module):
    """U-Net producing a segmentation logit map and, optionally, an orientation
    field.

    ``orientation`` is returned L2-normalised per pixel so that it is a genuine
    unit director; the loss then reduces to a cosine distance and the
    aggregation in ``geometry.orientation_from_field`` can treat every pixel as
    an equal-magnitude vote weighted only by segmentation confidence.
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        pretrained: bool = True,
        orientation_head: bool = True,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.encoder = ResNetEncoder(encoder, pretrained)
        self.has_orientation = orientation_head

        enc = self.encoder.channels
        skips = (enc[3], enc[2], enc[1], enc[0], 0)
        stages = []
        in_channels = enc[4]
        for out_channels, skip in zip(decoder_channels, skips):
            stages.append(_DecoderStage(in_channels, skip, out_channels))
            in_channels = out_channels
        self.decoder = nn.ModuleList(stages)

        self.mask_head = nn.Conv2d(decoder_channels[-1], 1, 1)
        self.orientation_head = (
            nn.Conv2d(decoder_channels[-1], 2, 1) if orientation_head else None
        )

    def forward(self, x):
        features = self.encoder(x)
        skips = [features[3], features[2], features[1], features[0], None]
        out = features[4]
        for stage, skip in zip(self.decoder, skips):
            out = stage(out, skip)

        result = {"mask": self.mask_head(out)}
        if self.orientation_head is not None:
            orientation = self.orientation_head(out)
            result["orientation"] = orientation / (
                orientation.norm(dim=1, keepdim=True) + 1e-6
            )
        return result


def build_model(config) -> GeometryAwareUNet:
    return GeometryAwareUNet(
        encoder=config.encoder,
        pretrained=config.encoder_weights == "imagenet",
        orientation_head=config.uses_orientation,
    )
