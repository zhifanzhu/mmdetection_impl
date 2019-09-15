import logging

import math
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class ExtraConv(nn.Module):
    def __init__(self, inp, outp, stride, insert_1x1_conv, min_depth=16):
        super(ExtraConv, self).__init__()
        if insert_1x1_conv:
            outp_1x1 = max(min_depth, outp // 2)
            layers = [
                nn.Conv2d(inp, outp_1x1, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(outp_1x1, outp, kernel_size=3, stride=stride, padding=1),
                nn.ReLU6(inplace=True),
            ]
        else:
            layers = [
                nn.Conv2d(inp, outp, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True)
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


@BACKBONES.register_module
class SSDMobileNetV2(MobileNetV2):
    def __init__(self,
                 input_size,
                 google_stype=True):
        super(SSDMobileNetV2, self).__init__(
            n_class=1,
            width_mult=1.0)
        assert input_size == 300
        self.input_size = input_size

        if google_stype:
            self.extra = nn.ModuleList([
                ExtraConv(1280, 512, stride=2, insert_1x1_conv=True),
                ExtraConv(512, 256, stride=2, insert_1x1_conv=True),
                ExtraConv(256, 256, stride=2, insert_1x1_conv=True),
                ExtraConv(256, 128, stride=2, insert_1x1_conv=True),
            ])
        else:
            self.extra = nn.ModuleList([
                InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
                InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
                InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
                InvertedResidual(256, 128, stride=2, expand_ratio=0.25)
            ])

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            # layer15/expansion_output
            if i == 14:
                for i_sub, conv_op in enumerate(layer.conv):
                    x = conv_op(x)
                    if i_sub == 0:
                        outs.append(x)
                continue
            # layer19
            if i == 18:
                x = layer(x)
                outs.append(x)
                continue

            x = layer(x)

        for i, layer in enumerate(self.extra):
            x = layer(x)
            outs.append(x)

        return tuple(outs)
