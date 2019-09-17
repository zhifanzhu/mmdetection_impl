from itertools import product
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.ssd_mobilenet_v2 import ExtraConv, InvertedResidual

from ..registry import TEMPORAL_MODULE


@TEMPORAL_MODULE.register_module
class SimpleConcat(nn.Module):
    def __init__(self,
                 in_channels=(576, 1280, 512, 256, 256, 128),
                 start_level=1,
                 ssd_pth_path=None,
                 ):
        super(SimpleConcat, self).__init__()
        self.in_channels = in_channels
        self.start_level = start_level
        assert start_level >= 1
        # self.decoder = nn.Sequential()  # Identity Module

        self.extra = nn.ModuleList([
            ExtraConv(
                in_channels[i-1],
                in_channels[i],
                stride=2,
                insert_1x1_conv=True)
            for i in range(start_level + 1, len(in_channels))
        ])

        self.concat_conv = InvertedResidual(
            inp=in_channels[start_level] * 2,
            oup=in_channels[start_level],
            stride=1,
            expand_ratio=1)

        self.import_extra_weights(ssd_pth_path)

    def import_extra_weights(self, state_dict_path=None):
        # Hack: assume first two of in_channels is not in `extra`
        # and, assume extra use google's style.
        if state_dict_path is not None:
            print(f'loading extra weights from {state_dict_path}...')
            state_dict = torch.load(state_dict_path)['state_dict']
            for i in range(self.start_level + 1, len(self.in_channels)):
                for ind, m in product((0, 1), ('weight', 'bias')):
                    src_key = f'backbone.extra.{i-2}.conv.{ind}.{m}'
                    dst_key = f'{i-2}.conv.{ind}.{m}'
                    self.extra.state_dict()[dst_key].data.copy_(
                        state_dict[src_key])

    def init_weights(self):
        for m in self.concat_conv.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs_list, in_dict=None, is_train=False):
        # time, batch, c0, h0, w0 = inputs_list[0].shape
        outs = []

        for i in range(0, self.start_level):
            t_i, b_i, c_i, h_i, w_i = inputs_list[i].shape
            outs.append(inputs_list[i].view(t_i*b_i, c_i, h_i, w_i))

        inputs_base = inputs_list[self.start_level]
        base_feats = [inputs_base[0]]  # First frame
        for t in range(1, len(inputs_base)):
            cat_feat = torch.cat([inputs_base[t-1], inputs_base[t]], dim=1)
            cat_feat = F.relu6(self.concat_conv(cat_feat))
            base_feats.append(cat_feat)
        base_feats = torch.cat(base_feats, dim=0)  # stack then reshape T*B
        outs.append(base_feats)

        feats = base_feats
        extra_level = 0
        for i in range(self.start_level + 1, len(self.in_channels)):
            feats = F.relu6(self.extra[extra_level](feats))
            extra_level += 1
            outs.append(feats)

        return outs, None
