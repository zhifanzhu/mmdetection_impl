import torch
import torch.nn as nn
from mmcv.cnn import xavier_init

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class Grab(nn.Module):

    def __init__(self, channels=256):
        super(Grab, self).__init__()
        self.conv_l = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            stride=2,
            activation='relu')
        self.conv_h = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            stride=1,
            activation='relu')
        self.conv_2 = nn.Sequential(
            ConvModule(
                in_channels=2*channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu'))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, f, f_h, f_l):
        f_l = self.conv_l(f_l)
        f_h = self.conv_h(f_h)
        f_prev = torch.cat([f_h, f_l], dim=1)
        f_prev = self.conv_2(f_prev)
        return f + f_prev


@PAIR_MODULE.register_module
class TwinGrab(nn.Module):

    def __init__(self, channels=256):
        super(TwinGrab, self).__init__()
        self.grabs = nn.ModuleList([Grab(channels=channels) for _ in range(4)])
        self.conv_extra = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            stride=1,
            activation='relu')

    def init_weights(self):
        for g in self.grabs:
            g.init_weights()
        for m in self.conv_extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feat, feat_ref, is_train=False):
        outs = [
            self.grabs[0](f=feat[0], f_h=feat_ref[1], f_l=feat_ref[0]),
            self.grabs[1](f=feat[1], f_h=feat_ref[2], f_l=feat_ref[1]),
            self.grabs[2](f=feat[2], f_h=feat_ref[3], f_l=feat_ref[2]),
            self.grabs[3](f=feat[3], f_h=feat_ref[4], f_l=feat_ref[3]),
            self.conv_extra(feat[4])
        ]
        return outs
