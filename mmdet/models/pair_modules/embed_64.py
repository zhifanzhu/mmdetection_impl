import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from ..utils import ConvModule

from ..registry import PAIR_MODULE

"""
Simplified from TEMPORAL_MODULE.
This file is a demonstration.
"""


@PAIR_MODULE.register_module
class Embed64(nn.Module):
    """ Identity temporal module, i.e. no modification on input data.
    """

    def __init__(self):
        super(Embed64, self).__init__()
        self.embed_conv = ConvModule(
            256,
            256,
            1,
            conv_cfg=None,
            norm_cfg=None,
            activation='relu',
            inplace=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feat, feat_ref, is_train=False):
        out = [
            self.embed_conv(feat[0]),
            feat[1],
            feat[2],
            feat[3],
            feat[4],
        ]
        return out
