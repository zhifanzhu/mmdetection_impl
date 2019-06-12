#
# ktw361 @ 2019.6.4
#

__author__ = 'ktw361'

import logging

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init, truncated_normal_init
from mmcv.runner import load_checkpoint

from ..backbones import ResNet
from ..backbones.resnet import make_res_layer, BasicBlock, Bottleneck
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

'''
We only use C4(layer3) as feature map, following speed/accuracy trade-off paper.
With 512 input, output feature map shape:
    [torch.Size([1, 1024, 32, 32]),
     torch.Size([1, 512, 16, 16]),
     torch.Size([1, 512, 8, 8]),
     torch.Size([1, 256, 4, 4]),
     torch.Size([1, 256, 2, 2]),
     torch.Size([1, 128, 1, 1])]

If C3 is also in atrous mode:
    out_from=('layer3', '', '', '', '', '', ''),
    out_channels=(-1, 512, 512, 256, 256, 128, 128),

ResNet-101:
plain conv in extra layer:
    Forward/backward pass size (MB): 2460.58
    Params size (MB): 215.01
use BasicBlock in extra layer:
    Forward/backward pass size (MB): 2469.61
    Params size (MB): 243.98
'''

STAGES_DEFAULT = (1, 2, 2, 2)
DILATIONS_DEFAULT = (1, 1, 1, 1)
OUT_INDICES_DEFAULT = (0, 1, 2, 3)
STAGE_WITH_DCN_DEFAULT = (False, False, False, False)

@BACKBONES.register_module
class SSDResNet(ResNet):

    def __init__(self,
                 depth=101,
                 num_stages=3,
                 out_from=('layer3', '', '', '', '', ''),
                 out_channels=(-1, 512, 512, 256, 256, 128),
                 use_dilation_conv4=False,
                 use_dilation_conv5=True,
                 use_resblock_in_extra=False,
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(SSDResNet, self).__init__(
            depth=depth,
            num_stages=num_stages,
            strides=STAGES_DEFAULT[:num_stages],
            dilations=DILATIONS_DEFAULT[:num_stages],
            out_indices=OUT_INDICES_DEFAULT[:num_stages],
            style=style,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=STAGE_WITH_DCN_DEFAULT[:num_stages],
            with_cp=with_cp,
            zero_init_residual=zero_init_residual)

        assert len(out_channels) == len(out_from)
        if use_dilation_conv4:
            assert num_stages == 2 and use_dilation_conv5, \
                'if use atrous on conv4_1, num_stages should be 2'
            assert len(out_channels) == 7, \
                'if use atrous on conv4_1, need to indicate one more feature map'
        else:
            if use_dilation_conv5:
                assert num_stages == 3, \
                    'if use atrous on conv5_1, num_stages should be 3'
            else:
                assert num_stages == 4, \
                    'if not using atrous, num_stages should be 4'

        self.use_resblock_in_extra = use_resblock_in_extra
        self.out_from = out_from
        self.out_channels = out_channels

        dcn = self.dcn if self.stage_with_dcn[-1] else None
        # 1. add atrous convs to C4
        if use_dilation_conv4:
            inplanes = self.feat_dim
            planes = 64 * 2 ** num_stages
            num_blocks = self.arch_settings[self.depth][1][num_stages]  # see arch_setting
            stride = 1
            dilation_first = 2
            dilation_other = 1
            res_layer = _make_res_atrous_layer(
                self.block,
                inplanes,
                planes,
                blocks=num_blocks,
                stride=stride,
                dilation_first=dilation_first,
                dilation_other=dilation_other,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn
            )
            layer_name = 'layer{}'.format(num_stages + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            self.feat_dim = self.block.expansion * 64 * 2**(num_stages)
            num_stages += 1

        # 2. add atrous conv as C5
        if use_dilation_conv5:
            inplanes = self.feat_dim
            planes = 64 * 2 ** num_stages
            num_blocks = self.arch_settings[self.depth][1][num_stages]
            stride = 1
            dilation_first = 2
            dilation_other = 1
            res_layer = _make_res_atrous_layer(
                self.block,
                inplanes,
                planes,
                blocks=num_blocks,
                stride=stride,
                dilation_first=dilation_first,
                dilation_other=dilation_other,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn
            )
            layer_name = 'layer{}'.format(num_stages + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            self.feat_dim = self.block.expansion * 64 * 2**(num_stages)

        # 3. add extra layers
        self.out_channels = out_channels
        self.extra = self._make_extra_layers(self.out_channels)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

        # init extra layer with truncated normal distribution
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                truncated_normal_init(m, mean=0, std=0.03)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if layer_name in self.out_from:
                outs.append(x)

        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            outs.append(x)

        return tuple(outs)

    def _make_extra_layers(self, outplanes):
        inplanes = self.feat_dim
        # inplanes = self.block.expansion * 64 * 2 ** self.num_stages  # 2048 if num_stages = 3
        layers = []
        for planes in outplanes:
            if planes == -1:
                continue
            if self.use_resblock_in_extra:
                extra_layer = make_res_layer(BasicBlock,
                                             inplanes,
                                             planes,
                                             blocks=1,
                                             stride=2,
                                             dilation=1
                                             )
            else:
                conv = build_conv_layer(self.conv_cfg,
                                        inplanes,
                                        planes,
                                        3,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        bias=False
                                        )
                norm_name, norm = build_norm_layer(self.norm_cfg, planes)
                extra_layer = nn.Sequential(conv, norm)
            layers.append(extra_layer)
            inplanes = planes
        return nn.Sequential(*layers)


def _make_res_atrous_layer(block,
                           inplanes,
                           planes,
                           blocks,
                           stride=1,
                           dilation_first=2,
                           dilation_other=1,
                           style='pytorch',
                           with_cp=False,
                           conv_cfg=None,
                           norm_cfg=dict(type='BN'),
                           dcn=None):
    downsample = nn.Sequential(
        build_conv_layer(
            conv_cfg,
            inplanes,
            planes * block.expansion,
            kernel_size=1,
            stride=stride,
            bias=False),
        build_norm_layer(norm_cfg, planes * block.expansion)[1],
    )
    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation_first,
            downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation_other,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn))

    return nn.Sequential(*layers)
