import torch
import torch.nn as nn
from mmcv.cnn import xavier_init

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class Direct(nn.Module):

    def __init__(self, use_skip=False, channels=256, bare=False, force_final=False):
        """

        :param use_skip:
        :param channels:
        :param bare: bool, if True, do not perform conv_h and conv_2,
            i.e. transform x_ref by conv_final and skip connect to x directly.
        :param force_final: if True, conv_final output as feature, rather than weight map
        """
        super(Direct, self).__init__()
        self.use_skip = use_skip
        self.bare = bare
        if not self.bare:
            self.conv_h = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=1)
            self.conv_2 = nn.Sequential(
                ConvModule(
                    in_channels=channels,
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
        self.force_final = force_final
        if self.force_final:
            chans = [channels, channels, channels, channels]
        else:
            final_chan = 1 if self.use_skip else 2
            chans = [256, 16, 3, final_chan]
        self.conv_final = nn.Sequential(
            ConvModule(
                in_channels=2*channels,
                out_channels=chans[0],
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[0],
                out_channels=chans[1],
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[1],
                out_channels=chans[2],
                kernel_size=1,
                padding=0,
                stride=1),
            nn.Conv2d(
                in_channels=chans[2],
                out_channels=chans[3],
                kernel_size=1,
                padding=0,
                stride=1)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        if not self.force_final:
            if self.use_skip:
                nn.init.constant_(self.conv_final[-1].bias, 0.0)
            else:
                nn.init.constant_(self.conv_final[-1].bias[0], 1.0)
                nn.init.constant_(self.conv_final[-1].bias[1], 0.0)

    def forward(self, f, f_h):
        if self.bare:
            f_prev = f_h
        else:
            f_h = self.conv_h(f_h)
            f_prev = self.conv_2(f_h)

        cat_feat = torch.cat([f, f_prev], dim=1)
        if self.use_skip:
            if self.force_final:
                out = self.conv_final(cat_feat)
            else:
                out = f + self.conv_final(cat_feat) * f_prev
        else:
            score = torch.softmax(self.conv_2(cat_feat), dim=1)
            out = score[:, 0, :, :].unsqueeze(1) * f + \
                    score[:, 1, :, :].unsqueeze(1) * f_prev
        return out


@PAIR_MODULE.register_module
class TwinDirectK1(nn.Module):

    def __init__(self,
                 use_skip=False,
                 channels=256,
                 bare=False,
                 top_conv=False,
                 shared=False,
                 force_final=False):
        super(TwinDirectK1, self).__init__()
        self.shared = shared
        if not shared:
            self.grabs = nn.ModuleList([
                Direct(
                    use_skip=use_skip, channels=channels, bare=bare, force_final=force_final)
                for _ in range(4)])
        else:
            self.grab = Direct(use_skip=use_skip, channels=channels, bare=bare)
        self.top_conv = top_conv
        if self.top_conv:
            self.conv_extra = ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,           # Note different stride
                    activation='relu')
            if not self.shared:
                self.grab_extra = Direct(
                        use_skip=use_skip,
                        channels=channels,
                        bare=bare)
        else:
            self.conv_extra = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=1,               # Note different stride
                activation='relu')

    def init_weights(self):
        if self.shared:
            self.grab.init_weights()
        else:
            for g in self.grabs:
                g.init_weights()
        for m in self.conv_extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        if self.top_conv and not self.shared:
            self.grab_extra.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        if self.shared:
            outs = [
                self.grab(f=feat[0], f_h=feat_ref[1]),
                self.grab(f=feat[1], f_h=feat_ref[2]),
                self.grab(f=feat[2], f_h=feat_ref[3]),
                self.grab(f=feat[3], f_h=feat_ref[4]),
            ]
        else:
            outs = [
                self.grabs[0](f=feat[0], f_h=feat_ref[1]),
                self.grabs[1](f=feat[1], f_h=feat_ref[2]),
                self.grabs[2](f=feat[2], f_h=feat_ref[3]),
                self.grabs[3](f=feat[3], f_h=feat_ref[4]),
            ]
        if self.top_conv:
            feat_ref_top = self.conv_extra(feat_ref[4])
            if self.shared:
                outs.append(self.grab(f=feat[4], f_h=feat_ref_top))
            else:
                outs.append(self.grab_extra(f=feat[4], f_h=feat_ref_top))
        else:
            outs.append(self.conv_extra(feat[4]))

        return outs
