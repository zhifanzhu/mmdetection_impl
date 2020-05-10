import torch
import torch.nn as nn
from mmcv.cnn import xavier_init

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class Grab(nn.Module):

    def __init__(self,
                 use_skip=False,
                 channels=256,
                 low_only=False,
                 dilation=False):
        super(Grab, self).__init__()
        self.use_skip = use_skip
        self.low_only = low_only
        conv_l_pad = 2 if dilation else 1
        conv_l_dilate = 2 if dilation else 1
        self.conv_l = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=conv_l_pad,
            dilation=conv_l_dilate,
            stride=2)
        if not self.low_only:
            self.conv_h = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=1)
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

    def forward(self, f, f_h, f_l):
        if self.low_only:
            f_prev = self.conv_l(f_l)
        else:
            f_l = self.conv_l(f_l)
            f_h = self.conv_h(f_h)
            f_prev = self.conv_2(torch.cat([f_h, f_l], dim=1))

        cat_feat = torch.cat([f, f_prev], dim=1)
        if self.use_skip:
            out = f + self.conv_final(cat_feat) * f_prev
        else:
            score = torch.softmax(self.conv_2(cat_feat), dim=1)
            out = score[:, 0, :, :].unsqueeze(1) * f + \
                    score[:, 1, :, :].unsqueeze(1) * f_prev
        return out


@PAIR_MODULE.register_module
class TwinGrab(nn.Module):

    def __init__(self,
                 use_skip=False,
                 channels=256,
                 low_only=False,
                 dilation=False,
                 shared=False,
                 extra_nls=False):
        super(TwinGrab, self).__init__()
        self.low_only = low_only
        self.shared = shared
        if not self.shared:
            self.grabs = nn.ModuleList(
                [Grab(use_skip=use_skip, channels=channels,
                      low_only=low_only, dilation=dilation)
                 for _ in range(4)])
        else:
            self.grab = Grab(use_skip=use_skip, channels=channels,
                             low_only=low_only, dilation=dilation)
        if self.low_only:
            if not self.shared:
                self.conv_extra = Grab(use_skip=use_skip, channels=channels,
                                       low_only=low_only, dilation=dilation)
        else:
            self.conv_extra = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu')

        if extra_nls:
            from .twin_nonlocal import NonLocal2D
            self.extra_nls = nn.ModuleList([
                NonLocal2D(
                    in_channels=channels,
                    reduction=2,
                    use_scale=True,
                    mode='embedded_gaussian',
                    conv_final=True)
                for _ in range(5)])

    def init_weights(self):
        if self.shared:
            self.grab.init_weights()
        else:
            for g in self.grabs:
                g.init_weights()
        if hasattr(self, 'conv_extra'):
            for m in self.conv_extra.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        if hasattr(self, 'extra_nls'):
            for g in self.extra_nls:
                g.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        if self.shared:
            outs = [
                self.grab(f=feat[0], f_h=feat_ref[1], f_l=feat_ref[0]),
                self.grab(f=feat[1], f_h=feat_ref[2], f_l=feat_ref[1]),
                self.grab(f=feat[2], f_h=feat_ref[3], f_l=feat_ref[2]),
                self.grab(f=feat[3], f_h=feat_ref[4], f_l=feat_ref[3]),
            ]
        else:
            outs = [
                self.grabs[0](f=feat[0], f_h=feat_ref[1], f_l=feat_ref[0]),
                self.grabs[1](f=feat[1], f_h=feat_ref[2], f_l=feat_ref[1]),
                self.grabs[2](f=feat[2], f_h=feat_ref[3], f_l=feat_ref[2]),
                self.grabs[3](f=feat[3], f_h=feat_ref[4], f_l=feat_ref[3]),
            ]
        if self.low_only:
            if self.shared:
                last_feat = self.grab(f=feat[4], f_h=None, f_l=feat_ref[4])
            else:
                last_feat = self.conv_extra(f=feat[4], f_h=None, f_l=feat_ref[4])
        else:
            last_feat = self.conv_extra(feat[4])
        outs.append(last_feat)

        if hasattr(self, 'extra_nls'):
            nl_outs = [
                self.extra_nls[0](x=outs[0], x_ref=feat_ref[1]),
                self.extra_nls[1](x=outs[1], x_ref=feat_ref[2]),
                self.extra_nls[2](x=outs[2], x_ref=feat_ref[3]),
                self.extra_nls[3](x=outs[3], x_ref=feat_ref[4]),
                self.extra_nls[4](x=outs[4], x_ref=outs[4]),
            ]
            return nl_outs
        else:
            return outs
