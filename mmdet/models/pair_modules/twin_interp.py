import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class Interp(nn.Module):

    def __init__(self,
                 use_skip=False,
                 channels=256):
        super(Interp, self).__init__()
        self.use_skip = use_skip
        self.conv_2 = nn.Sequential(
            ConvModule(
                in_channels=2*channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,
                stride=1),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=1)
        )
        final_chan = 1 if self.use_skip else 2
        self.conv_final = nn.Sequential(
            ConvModule(
                in_channels=2*channels,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=256,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.Conv2d(
                in_channels=3,
                out_channels=final_chan,
                kernel_size=3,
                padding=1,
                stride=1)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        if self.use_skip:
            nn.init.constant_(self.conv_final[-1].bias, 0.0)
        else:
            nn.init.constant_(self.conv_final[-1].bias[0], 1.0)
            nn.init.constant_(self.conv_final[-1].bias[1], 0.0)

    def forward(self, f, f_l):
        h, w = f.size(2), f.size(3)
        f_prev = F.interpolate(f_l, (h, w), mode='bilinear')

        cat_feat = torch.cat([f, f_prev], dim=1)
        if self.use_skip:
            out = f + self.conv_final(cat_feat) * f_prev
        else:
            score = torch.softmax(self.conv_2(cat_feat), dim=1)
            out = score[:, 0, :, :].unsqueeze(1) * f + \
                    score[:, 1, :, :].unsqueeze(1) * f_prev
        return out


@PAIR_MODULE.register_module
class TwinInterp(nn.Module):

    def __init__(self, use_skip=False, channels=256):
        super(TwinInterp, self).__init__()
        self.interps = nn.ModuleList(
            [Interp(use_skip=use_skip, channels=channels)
             for _ in range(5)])

    def init_weights(self):
        for g in self.interps:
            g.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        outs = [
            self.interps[0](f=feat[0], f_l=feat_ref[0]),
            self.interps[1](f=feat[1], f_l=feat_ref[1]),
            self.interps[2](f=feat[2], f_l=feat_ref[2]),
            self.interps[3](f=feat[3], f_l=feat_ref[3]),
            self.interps[4](f=feat[4], f_l=feat_ref[4]),
        ]

        return outs
