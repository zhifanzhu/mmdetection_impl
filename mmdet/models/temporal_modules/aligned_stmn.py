import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from mmcv.cnn import normal_init
from mmdet.ops import Correlation, DeformConv


"""
See stmn.py,
here we add *matchtrans* to it.
"""


class AlignedSTMNCell(nn.Module):
    """Basic STMN recurrent network cell.

    Now in_channels must equalt to hidden_size

    """

    def __init__(self,
                 in_channels,
                 hidden_size,
                 kernel_size=3,
                 displacement=4):
        super(AlignedSTMNCell, self).__init__()
        padding = kernel_size // 2
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(
            in_channels + hidden_size,
            hidden_size,
            kernel_size,
            padding=padding)
        self.update_gate = nn.Conv2d(
            in_channels + hidden_size,
            hidden_size,
            kernel_size,
            padding=padding)
        self.out_gate = nn.Conv2d(
            in_channels + hidden_size,
            hidden_size,
            kernel_size,
            padding=padding)

        # bottleneck = 256
        offset_channels = kernel_size * kernel_size * 2
        corr_channels = (2 * displacement + 1) ** 2
        self.correlation = Correlation(
            pad_size=displacement, kernel_size=1, max_displacement=displacement,
            stride1=1, stride2=1)
        # self.corr_conv = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=corr_channels,
        #         out_channels=bottleneck,
        #         kernel_size=kernel_size,
        #         padding=(kernel_size - 1) // 2,
        #         stride=1),
        #     nn.ReLU(inplace=True))
        self.conv_offset = nn.Conv2d(
            corr_channels,
            offset_channels,
            kernel_size=1,
            bias=False)
        self.conv_align = DeformConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) //2,
            deformable_groups=1)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)
        # normal_init(self.corr_conv[0], std=0.01)
        normal_init(self.conv_offset, std=0.01)
        normal_init(self.conv_align, std=0.01)

    @staticmethod
    def _linear_scale(inputs, std_multiplier=3.0):
        if len((inputs > 0).nonzero()) > 0:
            inputs_flat = inputs.view(-1)
            pos_inputs = inputs_flat[inputs_flat > 0]
            mean = torch.mean(pos_inputs)
            std = torch.std(pos_inputs)
            if bool(torch.isnan(mean)) or bool(torch.isnan(std)):
                upper_bound = 1.0
            else:
                upper_bound = mean + std * std_multiplier
        else:
            upper_bound = 1.0

        inputs = torch.clamp(inputs, 0, float(upper_bound))
        inputs = inputs * (1.0 / upper_bound)
        return inputs

    def forward(self, inputs, in_state):
        # data size is [batch, channel, height, width]
        aligned_memory = self.align_features(prev_feat=in_state, cur_feat=inputs)
        stacked_inputs = torch.cat([inputs, aligned_memory], dim=1)
        update = self._linear_scale(F.relu(self.update_gate(stacked_inputs)))
        reset = self._linear_scale(F.relu(self.reset_gate(stacked_inputs)))
        out_inputs = F.relu(self.out_gate(torch.cat([inputs, in_state * reset], dim=1)))
        new_state = in_state * (1.0 - update) + out_inputs * update
        out_feat = new_state

        return out_feat, new_state

    def align_features(self, prev_feat, cur_feat):
        corr_feat = self.correlation(cur_feat, prev_feat)
        # corr_feat = self.corr_conv(corr_feat)
        offset = self.conv_offset(corr_feat)
        aligned_feat = self.conv_align(prev_feat, offset)
        return aligned_feat

    def init_state(self, in_shape, device):
        # get batch and spatial sizes
        batch_size = in_shape[0]
        spatial_size = in_shape[2:]

        # generate empty prev_state, if None is provided
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = torch.zeros(state_size, device=device)
        return state
