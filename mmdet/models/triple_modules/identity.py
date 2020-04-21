import torch.nn as nn

from ..registry import TRIPLE_MODULE

"""
Simplified from TEMPORAL_MODULE.
This file is a demonstration.
"""


@TRIPLE_MODULE.register_module
class Identity(nn.Module):
    """ Triple Identity temporal module, i.e. no modification on input data.
    """

    def __init__(self):
        super(Identity, self).__init__()
        self.decoder = nn.Sequential()  # Triple Identity Module

    def init_weights(self):
        pass

    def forward(self, feat, feat_near, feat_far, is_train=False):
        return feat
