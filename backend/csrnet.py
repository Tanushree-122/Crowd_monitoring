"""
csrnet.py
---------
Implementation of the CSRNet architecture (Congested Scene Recognition Network)
as described in:

    Li, Y., Zhang, X., & Chen, D. (2018).
    "CSRNet: Dilated Convolutional Neural Networks for Understanding the
    Highly Congested Scenes." CVPR 2018.

CSRNet is composed of two parts:
  1. Frontend: the first 10 convolutional layers of VGG-16 (used as a
     feature extractor). This gives the network strong, transferable
     low/mid level visual features.
  2. Backend: a stack of DILATED convolutions (instead of pooling/stride)
     which enlarges the receptive field without reducing spatial
     resolution, which is critical for producing high quality density
     maps for dense/congested crowds.

The final 1x1 convolution maps the backend feature maps to a single
channel density map. The predicted crowd count is obtained by summing
all pixel values in the density map (since the map is trained so that
the integral over the map equals the head count in that region).

NOTE ON WEIGHTS:
This file ONLY defines the architecture. It does NOT fabricate or invent
any pretrained weights. Actual pretrained weights compatible with this
exact architecture must be supplied by the user in:

    backend/models/csrnet.pth

See README.md for exact instructions on where to obtain compatible
CSRNet weights (trained on ShanghaiTech Part A/B or UCF-QNRF).
"""

import collections
import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    """
    CSRNet model.

    Args:
        load_weights (bool): If True, skips ImageNet-pretrained VGG16
            initialization for the frontend (used when you are about to
            immediately load a full CSRNet checkpoint anyway, which makes
            the ImageNet init pointless and saves a network call / time).
            If False, the frontend is initialized with ImageNet-pretrained
            VGG16 weights (standard CSRNet training procedure) -- this is
            only useful for TRAINING from scratch, not for our inference
            use-case, but is kept for completeness/fidelity to the paper.
    """

    def __init__(self, load_weights: bool = True):
        super(CSRNet, self).__init__()
        self.seen = 0

        # First 10 conv layers of VGG16 (frontend feature extractor)
        self.frontend_feat = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512
        ]

        # Dilated backend (dilation rate = 2, standard CSRNet config "B")
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat, in_channels=512, dilation=True
        )

        # Output layer: collapses 64 channels -> 1-channel density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            self._init_frontend_from_vgg16()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _init_frontend_from_vgg16(self):
        """Initialize frontend weights from torchvision's pretrained VGG16."""
        try:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            # Fallback for older torchvision versions
            vgg16 = models.vgg16(pretrained=True)

        self._initialize_new_weights()

        vgg16_state = list(vgg16.state_dict().items())
        frontend_state = self.frontend.state_dict()
        frontend_keys = list(frontend_state.keys())

        fsd = collections.OrderedDict()
        for i in range(len(frontend_keys)):
            fsd[frontend_keys[i]] = vgg16_state[i][1]

        self.frontend.load_state_dict(fsd)

    def _initialize_new_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels: int = 3, batch_norm: bool = False,
                 dilation: bool = False):
    """
    Build a sequential stack of Conv2d/ReLU (and optional BatchNorm/MaxPool)
    layers from a VGG-style configuration list.

    Args:
        cfg: list of ints (output channels) and 'M' (max-pool marker).
        in_channels: number of input channels to the first conv layer.
        batch_norm: whether to insert BatchNorm2d after each conv.
        dilation: if True, use dilated convolutions (dilation=2, padding=2)
                  instead of standard stride-1 padding-1 convs. This is
                  what CSRNet uses in its backend to expand the receptive
                  field without downsampling spatial resolution.
    """
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3,
                padding=d_rate, dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
