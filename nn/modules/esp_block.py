# ESPDet 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Block modules."""

import torch.nn as nn
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules import C2f, C3, Bottleneck
from .esp_conv import DSConv

__all__ = (
    "DSBottleneck",
    "DSC3k2",
    "ESPSerial",
    "ESPSerialLite",
    "ESPBlock",
    "ESPBlockLite",
)

class C3k(C3):
    """
    C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks.
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class DSBottleneck(nn.Module):
    """Replace Conv in standard bottleneck with DSConv."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a ds_bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DSConv(c1, c_, k[0], 1)
        self.cv2 = DSConv(c_, c2, k[1], 1) #g is removed
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DSC3(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DSBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class DSC3k2(C2f):
    """Replace the standard bottleneck in C3k2 with DSBottleneck."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else DSBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class ESPSerial(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): # True in pico
        """
        Initialize a ESPBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(n * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(2 * self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        x = [self.cv1(x)]
        x.extend(m(x[-1]) for m in self.m)
        return self.cv2(x[-1])


class ESPSerialLite(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        x = [self.cv1(x)]
        x.extend(m(x[-1]) for m in self.m)
        return self.cv2(x[-1])


class ESPBlock(ESPSerial):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(2 * self.c, self.c, 2, shortcut, g) if c3k else DSBottleneck(2 * self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class ESPBlockLite(ESPSerialLite):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=False):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else DSBottleneck(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )