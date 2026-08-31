# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import torch.nn as nn

__all__ = "DSConv"


class DSConv(nn.Module):  #depthwise separable conv
    """ Implementation of Depthwise Separable Convolution. """
    def __init__(self, c1, c2, k=3, s=1, p=1, dilation=1, groups=1, act=nn.ReLU(inplace=False)):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(c1, c1, k, s, p, groups=c1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pointwise = nn.Conv2d(c1, c2, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = act

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x



