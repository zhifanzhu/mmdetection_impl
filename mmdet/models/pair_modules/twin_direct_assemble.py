import torch
import torch.nn as nn
from mmdet.ops import MxAssemble
from mmcv.cnn import xavier_init

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class DirectAssemble(nn.Module):

    def __init__(self,
                 displacement,
                 use_skip=False,
                 channels=256):
        super(DirectAssemble, self).__init__()
        self.use_skip = use_skip
        self.disp = displacement
        self.radius = 2 * displacement + 1

        self.atten_net = None
        self.assemble = None
        if self.disp != 0:
            atten_chans = [256, 128, self.radius * self.radius]
            self.atten_net = nn.Sequential(
                ConvModule(
                    in_channels=2*channels,
                    out_channels=atten_chans[0],
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    activation='relu'),
                ConvModule(
                    in_channels=atten_chans[0],
                    out_channels=atten_chans[1],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    activation='relu'),
                ConvModule(
                    in_channels=atten_chans[1],
                    out_channels=atten_chans[2],
                    kernel_size=3,
                    padding=1,
                    stride=1))

            self.assemble = MxAssemble(k=self.disp)

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
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[1],
                out_channels=chans[2],
                kernel_size=3,
                padding=1,
                stride=1),
            nn.Conv2d(
                in_channels=chans[2],
                out_channels=chans[3],
                kernel_size=3,
                padding=1,
                stride=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        if self.use_skip:
            nn.init.constant_(self.conv_final[-1].bias, 0.0)
        else:
            nn.init.constant_(self.conv_final[-1].bias[0], 1.0)
            nn.init.constant_(self.conv_final[-1].bias[1], 0.0)

    def forward(self, f, f_h):
        if self.disp == 0:
            aligned = f_h
        else:
            atten_input = torch.cat([f, f_h], dim=1)
            atten = self.atten_net(atten_input)
            aligned = self.assemble(atten, f_h)

        cat_feat = torch.cat([f, aligned], dim=1)
        if self.use_skip:
            out = f + self.conv_final(cat_feat) * aligned
        else:
            score = torch.softmax(self.conv_final(cat_feat), dim=1)
            out = score[:, 0, :, :].unsqueeze(1) * f + \
                    score[:, 1, :, :].unsqueeze(1) * aligned
        return out


@PAIR_MODULE.register_module
class TwinDirectAssemble(nn.Module):

    def __init__(self,
                 disp_list=(4, 2, 1, 0, 0),
                 use_skip=False,
                 channels=256):
        super(TwinDirectAssemble, self).__init__()
        self.disp_list = disp_list
        self.grabs = nn.ModuleList([
            DirectAssemble(
                displacement=disp_list[i],
                use_skip=use_skip,
                channels=channels)
            for i in range(4)])

        self.conv_extra = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=2,           # Note different stride
                activation='relu')
        self.grab_extra = DirectAssemble(
            displacement=disp_list[4],
            use_skip=use_skip,
            channels=channels)

    def init_weights(self):
        for g in self.grabs:
            g.init_weights()
        self.grab_extra.init_weights()
        for m in self.conv_extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feat, feat_ref, is_train=False):
        outs = [
            self.grabs[0](f=feat[0], f_h=feat_ref[1]),
            self.grabs[1](f=feat[1], f_h=feat_ref[2]),
            self.grabs[2](f=feat[2], f_h=feat_ref[3]),
            self.grabs[3](f=feat[3], f_h=feat_ref[4]),
        ]
        feat_ref_top = self.conv_extra(feat_ref[4])
        outs.append(self.grab_extra(f=feat[4], f_h=feat_ref_top))

        return outs
