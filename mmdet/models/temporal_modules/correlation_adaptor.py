import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.ops import PointwiseCorrelation, DeformConv

from ..registry import TEMPORAL_MODULE

"""
[torch.Size([1, 256, 64, 64]),
 torch.Size([1, 256, 32, 32]),
 torch.Size([1, 256, 16, 16]),
 torch.Size([1, 256, 8, 8]),
 torch.Size([1, 256, 4, 4])]
 
Only perform corr on first 4 layers,
independent corr, they don't affect other layer.
"""


@TEMPORAL_MODULE.register_module
class CorrelationAdaptor(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 displacements=(8, 8, 4, 2),
                 strides=(2, 1, 1, 1),
                 kernel_size=3,
                 deformable_groups=4):
        super(CorrelationAdaptor, self).__init__()
        assert len(displacements) == len(strides)
        offset_channels = kernel_size * kernel_size * 2
        self.point_corrs = nn.ModuleList([
            PointwiseCorrelation(disp, s)
            for disp, s in zip(displacements, strides)
        ])
        self.conv_offsets = nn.ModuleList([
            nn.Conv2d(
                (2*disp+1)*(2*disp+1),
                deformable_groups * offset_channels,
                kernel_size=1,
                bias=False)
            for disp in displacements
        ])
        self.conv_adaptions = nn.ModuleList([
            DeformConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                deformable_groups=deformable_groups)
            for _ in displacements
        ])
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for conv_offset, conv_adaption in zip(
                self.conv_offsets, self.conv_adaptions):
            normal_init(conv_offset, std=0.01)
            normal_init(conv_adaption, std=0.01)

    # TODO(zhifan) use previous in_dict during test.
    def forward(self, input_list, in_dict=None, is_train=False):
        outs = []
        num_adapt_layers = len(self.conv_offsets)
        for level, inputs in enumerate(input_list[:num_adapt_layers]):
            feats = self.forward_single(inputs, level)
            outs.append(feats)
        for inputs in input_list[num_adapt_layers:]:
            time, batch, c, h, w = inputs.shape
            outs.append(inputs.view([time*batch, c, h, w]))
        if is_train:
            return outs, None
        else:
            # return inputs for reuse in next frame/clips
            out_dict = dict(
                inputs=input_list
            )
            return outs, out_dict

    def forward_single(self, inputs, level):
        time, batch, c, h, w = inputs.shape
        if time <= 1:
            return inputs

        corr_feats = []
        for t in range(1, len(inputs)):
            # corr_feat = self.point_corrs[level](inputs[t-1], inputs[t])
            corr_feat = self.point_corrs[level](inputs[t], inputs[t-1])  # changed on 09.14
            corr_feat = corr_feat.reshape(batch, h, w, -1).permute(0, 3, 1, 2)
            corr_feats.append(corr_feat)
        # Parallelize offset and dconv computation, though no speedup.
        corr_feats = torch.cat(corr_feats)
        offset = self.conv_offsets[level](corr_feats)
        feat_inputs = inputs[1:].view((time - 1) * batch, c, h, w)
        feats = self.relu(self.conv_adaptions[level](feat_inputs, offset))
        feats = torch.cat([inputs[0], feats], dim=0)
        return feats
