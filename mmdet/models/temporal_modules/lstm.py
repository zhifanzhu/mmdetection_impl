import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init

from ..registry import TEMPORAL_MODULE


class BottleneckLSTMCell(nn.Module):
    """Basic LSTM recurrent network cell using separable convolutions.

    The implementation is based on:
    Mobile Video Object Detection with Temporally-Aware Feature Maps
    https://arxiv.org/abs/1711.06368.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    This LSTM first projects inputs to the size of the output before doing gate
    computations. This saves params unless the input is less than a third of the
    state size channel-wise.
    """

    def __init__(self,
                 in_channels,
                 hidden_size,
                 kernel_size=3,
                 forget_bias=1.0,
                 activation='relu6',
                 clip_state=False):
        super(BottleneckLSTMCell, self).__init__()
        assert hasattr(F, activation)
        self._hidden_size = hidden_size
        self._forget_bias = forget_bias
        self.activation = getattr(F, activation)
        self._clip_state = clip_state

        bottleneck_in_channels = in_channels + hidden_size
        self.bottleneck_conv_depth = nn.Conv2d(
            in_channels=bottleneck_in_channels,
            out_channels=bottleneck_in_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=bottleneck_in_channels)
        self.bottleneck_conv_point = nn.Conv2d(
            in_channels=bottleneck_in_channels,
            out_channels=hidden_size,
            kernel_size=1)

        self.lstm_conv_depth = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=hidden_size)
        self.lstm_conv_point = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=4 * hidden_size,
            kernel_size=1)

    def init_weights(self):
        xavier_init(self.bottleneck_conv_depth)
        xavier_init(self.bottleneck_conv_point)
        xavier_init(self.lstm_conv_depth)
        xavier_init(self.lstm_conv_point)

    def forward(self, inputs, in_state):
        cell = in_state['cell']
        hidden = in_state['hidden']

        bottleneck = self.bottleneck_conv_depth(
            torch.cat([inputs, hidden], dim=1))
        bottleneck = self.bottleneck_conv_point(bottleneck)

        concat = self.lstm_conv_depth(bottleneck)
        concat = self.lstm_conv_point(concat)

        i, j, f, o = torch.chunk(concat, 4, dim=1)

        new_cell = (
            cell * torch.sigmoid(f + self._forget_bias) +
            torch.sigmoid(i) * self.activation(j))
        if self._clip_state:
            new_cell = torch.clamp(new_cell, -6.0, 6.0)
        new_hidden = self.activation(new_cell) * torch.sigmoid(o)

        output = new_hidden
        # if self._output_bottleneck:

        out_dict = dict(
            cell=new_cell,
            hidden=new_hidden)
        return output, out_dict

    def init_state(self, in_shape, device):
        batch, chan, height, width = in_shape
        cell = torch.zeros(
            [batch, self._hidden_size, height, width],
            device=device,
            requires_grad=True)
        hidden = torch.zeros(
            [batch, self._hidden_size, height, width],
            device=device,
            requires_grad=True)
        return dict(
            cell=cell,
            hidden=hidden)


@TEMPORAL_MODULE.register_module
class BottleneckLSTMDecoder(nn.Module):
    """ Apply BottleneckLSTM cell to ONLY ONE layer in feature map.
    Future existed layers (with smaller size) will be replaced by convolution on
    previous layer after 'from_layer'.
        E.g.: , from_layer=2, in_channels [256, 256, 256, 256, 256],
        with shape [64, 32, 16, 8, 4],
        the result outputs will be [64, 32, 16_new, 8_new, 4_new].

    """

    def __init__(self,
                 in_channels,
                 lstm_cfg,
                 from_layer=2,
                 neck_first=True):
        """
        Args:
            in_channels: list of int, #-feat-channel of each feature layer
                during forward, this should match the real input.
            lstm_cfg: Config
            neck_first: bool
            from_layer: start from 0, to N-1, start LSTM from
                which layer of FPN/Backbone output.
        """
        super(BottleneckLSTMDecoder, self).__init__()
        assert from_layer > 0
        assert from_layer < len(in_channels)
        self.neck_first = neck_first
        self.from_layer = from_layer
        self.lstm_cell = BottleneckLSTMCell(**lstm_cfg)
        extra_convs = []
        for l in range(from_layer + 1, len(in_channels)):
            extra_convs.append(nn.Conv2d(
                in_channels=in_channels[l],
                out_channels=in_channels[l],
                kernel_size=3,
                stride=2,
                padding=1))
        self.extra_convs = nn.ModuleList(extra_convs)

    def init_weights(self):
        self.lstm_cell.init_weights()
        for conv in self.extra_convs:
            xavier_init(conv)

    def forward(self, input_list, in_dict=None, is_train=False):
        time, batch, chan, _, _ = input_list[0].shape

        final_outs = [
            inputs.permute(1, 0, 2, 3, 4).reshape([-1, *inputs.shape[2:]])
            for inputs in input_list[:self.from_layer]
        ]
        if is_train or in_dict is None:
            device = input_list[0].device
            _t, _b, _c, _h, _w = input_list[self.from_layer].shape
            state_dict = self.lstm_cell.init_state([_b, _c, _h, _w], device)
        else:
            state_dict = in_dict

        outs = []
        decoder_outs = []
        for decoder_input in input_list[self.from_layer]:
            feat, state_dict = self.lstm_cell(
                decoder_input, state_dict)
            decoder_outs.append(feat)
        decoder_outs = torch.cat(decoder_outs, dim=0)
        outs.append(decoder_outs)

        for conv in self.extra_convs:
            decoder_outs = F.relu(conv(decoder_outs))
            outs.append(decoder_outs)
        for out in outs:
            final_outs.append(
                out.reshape(
                    time, batch, *out.shape[1:]).permute(
                    1, 0, 2, 3, 4).reshape(
                    batch*time, *out.shape[1:]))
        return final_outs, state_dict
