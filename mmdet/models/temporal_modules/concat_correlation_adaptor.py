import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import DeformConv

from ..registry import TEMPORAL_MODULE


@TEMPORAL_MODULE.register_module
class ConcatCorrelationAdaptor(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 displacements=(4, 2),
                 strides=(2, 1),
                 kernel_size=3,
                 deformable_groups=4):
        """

        Args:
            in_channels: tuple of int
            out_channels: tuple of int, same length as in_channels
            displacements: tuple of int
            strides:tuple of int, same length as displacements
            kernel_size: int
            deformable_groups: int
        """
        super(ConcatCorrelationAdaptor, self).__init__()
        assert len(displacements) == len(strides)
        self.in_channels = in_channels
        self.out_channels = out_channels

        raise NotImplementedError
        self.point_corrs = nn.ModuleList([
            PointwiseCorrelation(disp, s)
            for disp, s in zip(displacements, strides)
        ])

        num_corr_channels = 0
        for d in displacements:
            num_corr_channels += (2*d+1) * (2*d+1)

        offset_channels = kernel_size * kernel_size * 2
        self.conv_offsets = nn.ModuleList([
            nn.Conv2d(
                num_corr_channels,
                deformable_groups * offset_channels,
                kernel_size=1,
                bias=False)
            for _ in in_channels
        ])
        self.conv_adaptions = nn.ModuleList([
            DeformConv(
                in_channels[i],
                out_channels[i],
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                deformable_groups=deformable_groups)
            for i, _ in enumerate(in_channels)
        ])
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for conv_offset, conv_adaption in zip(self.conv_offsets, self.conv_adaptions):
            normal_init(conv_offset, std=0.01)
            normal_init(conv_adaption, std=0.01)

    # TODO(zhifan) use previous in_dict during test.
    def forward(self, input_list, in_dict=None, is_train=False):
        lowest_shape = input_list[0].shape[-2:]
        outs = []

        num_corr_layers = len(self.point_corrs)
        corr_feats = []
        for level, inputs in enumerate(input_list[:num_corr_layers]):
            corr_feat = self.forward_single(inputs, level, lowest_shape)
            corr_feats.append(corr_feat)
        corr_feats = torch.cat(corr_feats, dim=1)

        num_adapt_layers = len(self.conv_adaptions)
        for level, inputs in enumerate(input_list[:num_adapt_layers]):
            time, batch, c, h, w = inputs.shape
            offset = self.conv_offsets[level](corr_feats)
            feats_input = input_list[level][1:].view(-1, c, h, w)
            feats = self.conv_adaptions[level](feats_input, offset)
            feats = torch.cat(
                [input_list[level][0, ...], feats], dim=0)
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

    def forward_single(self, inputs, level, lowest_shape):
        time, batch, c, h, w = inputs.shape
        if time <= 1:
            return inputs.view(-1, c, h, w)

        corr_feats = []
        for t in range(1, len(inputs)):
            if level > 0:
                input_t = F.interpolate(inputs[t], lowest_shape)
                input_t_1 = F.interpolate(inputs[t-1], lowest_shape)
            else:
                input_t = inputs[t]
                input_t_1 = inputs[t-1]
            corr_feat = self.point_corrs[level](input_t, input_t_1)
            corr_feat = corr_feat.view(
                batch, lowest_shape[0], lowest_shape[1], -1).permute(0, 3, 1, 2)
            corr_feats.append(corr_feat)
        # Parallelize offset and dconv computation, though no speedup.
        corr_feats = torch.cat(corr_feats, dim=0)
        return corr_feats
