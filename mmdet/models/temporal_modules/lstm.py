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
    """ Apply BottleneckLSTM cell to feature map.

    Future existed layers (with smaller size) may be replaced by convolution on
    previous layer.

    E.g.:
    1) out_layers_type=[0, 0, 1, 2, 2], in_channels=[256, 256, 256, 256, 256],
        with shape [64, 32, 16, 8, 4],
       the result outputs will be [64, 32, 16_new, 8_new, 4_new].
    2) [0, 0, 0, 0] means don't use any LSTM.
    3) [0, 1, 2, 2, 0] mean use old feature on first and last
       layer, while add one LSTM cell to second layer.

    """

    def __init__(self,
                 in_channels,
                 lstm_cfgs,
                 out_layers_type,
                 neck_first=True):
        """
        Args:
            in_channels: list of int, #-feat-channel of each feature layer
                during forward, this should match the real input.
            lstm_cfgs: list of Config, must match number of '1' in 'out_layers_type'
            out_layers_type: list of int, length should match in_channels.
                '0' means use old, '1' means use LSTM output,
                '2' means use Conv output of last '1' or '2' layer.
                Note '2' must proceed after '1' or '2', but not '0'.
            neck_first: bool
        """
        super(BottleneckLSTMDecoder, self).__init__()
        assert len(in_channels) == len(out_layers_type)
        self.neck_first = neck_first
        self.out_layers_type = out_layers_type
        self.num_olds = sum([1 for t in out_layers_type if t == 0])
        self.num_lstms = sum([1 for t in out_layers_type if t == 1])
        self.num_extra_convs = sum([1 for t in out_layers_type if t == 2])
        assert len(lstm_cfgs) == self.num_lstms

        if self.num_lstms:
            lstm_cells = []
            for i in range(self.num_lstms):
                lstm_cells.append(BottleneckLSTMCell(**lstm_cfgs[i]))
            self.lstm_cells = nn.ModuleList(lstm_cells)

        if self.num_extra_convs > 0:
            extra_convs = []
            for l, tp in enumerate(out_layers_type):
                if tp == 2:
                    extra_convs.append(nn.Conv2d(
                        in_channels=in_channels[l],
                        out_channels=in_channels[l],
                        kernel_size=3,
                        stride=2,
                        padding=1))
            self.extra_convs = nn.ModuleList(extra_convs)

    def init_weights(self):
        if self.num_lstms > 0:
            for lstm_cell in self.lstm_cells:
                lstm_cell.init_weights()
        if self.num_extra_convs > 0:
            for conv in self.extra_convs:
                xavier_init(conv)

    def forward(self, input_list, in_dict=None, is_train=False):
        # First process time dimension in lower layer, then go up
        # This will affect how state_dict updates.
        time, batch, chan, _, _ = input_list[0].shape

        if is_train or in_dict is None:
            device = input_list[0].device
            state_dict_lists = []
            lstm_lvl = 0
            for lvl, tp in enumerate(self.out_layers_type):
                if tp == 1:
                    _t, _b, _c, _h, _w = input_list[lvl].shape
                    in_shape = [_b, _c, _h, _w]
                    state_dict_lists.append(
                        self.lstm_cells[lstm_lvl].init_state(in_shape, device))
                    lstm_lvl += 1
            in_dict = dict(state_dict_lists=state_dict_lists)

        state_dict_lists = in_dict['state_dict_lists']

        final_outs = []
        out_dict_lists = []
        lstm_level = 0
        extra_level = 0
        feats = None
        for lvl, inputs in enumerate(input_list):
            if self.out_layers_type[lvl] == 0:
                # Use exists
                out = inputs.permute(1, 0, 2, 3, 4)
                out = out.reshape([-1, *inputs.shape[2:]])

            elif self.out_layers_type[lvl] == 1:
                # Use output of LSTM
                feats, state_dict = self.forward_lstm(
                    inputs,
                    self.lstm_cells[lstm_level],
                    state_dict=state_dict_lists[lstm_level])
                feats = feats.permute(1, 0, 2, 3, 4)
                feats = feats.reshape([-1, *feats.shape[2:]])
                out = feats
                lstm_level += 1
                out_dict_lists.append(state_dict)
            else:
                # Use last feature map.
                feats = F.relu(self.extra_convs[extra_level](feats))
                out = feats
                extra_level += 1
            final_outs.append(out)

        out_dict = dict(state_dict_lists=out_dict_lists)
        return final_outs, out_dict

    @staticmethod
    def forward_lstm(inputs, lstm_cell, state_dict):
        decoder_outs = []
        for decoder_input in inputs:
            feat, state_dict = lstm_cell(decoder_input, state_dict)
            decoder_outs.append(feat)
        decoder_outs = torch.stack(decoder_outs, dim=0)
        return decoder_outs, state_dict
