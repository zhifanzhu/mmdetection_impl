import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from mmcv.cnn import xavier_init

from ..registry import TEMPORAL_MODULE

"""
STMN is very similar to ConvGRU, except
1) use relu instead of sigmoid and tanh
2) use linear_scale after z and r, in paper they called it 
    BN_star, but it has no trainable parameter at all.
3) MatchTrans
4) Init weights using `swapped out conv layers`

We implemented 1) and 2).

"""

class STMNCell(nn.Module):
    """Basic STMN recurrent network cell.

    """

    def __init__(self,
                 in_channels,
                 hidden_size,
                 kernel_size=3):
        super(STMNCell, self).__init__()
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

    def init_weights(self):
        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)

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
        stacked_inputs = torch.cat([inputs, in_state], dim=1)
        update = self._linear_scale(F.relu(self.update_gate(stacked_inputs)))
        reset = self._linear_scale(F.relu(self.reset_gate(stacked_inputs)))
        out_inputs = F.relu(self.out_gate(torch.cat([inputs, in_state * reset], dim=1)))
        new_state = in_state * (1.0 - update) + out_inputs * update
        out_feat = new_state

        return out_feat, new_state

    def init_state(self, in_shape, device):
        # get batch and spatial sizes
        batch_size = in_shape[0]
        spatial_size = in_shape[2:]

        # generate empty prev_state, if None is provided
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        state = torch.zeros(state_size, device=device)
        return state


@TEMPORAL_MODULE.register_module
class RNNDecoder(nn.Module):
    """ Apply a specific rnn cell to feature map.

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
                 rnncell_type,
                 rnn_cfgs,
                 out_layers_type,
                 neck_first=True):
        """
        Args:
            in_channels: list of int, #-feat-channel of each feature layer
                during forward, this should match the real input.
            rnncell_type: str, name of Class
            rnn_cfgs: list of Config, must match number of '1' in 'out_layers_type'
            out_layers_type: list of int, length should match in_channels.
                '0' means use old, '1' means use LSTM output,
                '2' means use Conv output of last '1' or '2' layer.
                Note '2' must proceed after '1' or '2', but not '0'.
            neck_first: bool
        """
        super(RNNDecoder, self).__init__()
        assert len(in_channels) == len(out_layers_type)
        self.neck_first = neck_first
        self.out_layers_type = out_layers_type
        self.num_olds = sum([1 for t in out_layers_type if t == 0])
        self.num_rnns = sum([1 for t in out_layers_type if t == 1])
        self.num_extra_convs = sum([1 for t in out_layers_type if t == 2])
        assert rnncell_type in globals()
        assert len(rnn_cfgs) == self.num_rnns
        rnncell_obj = globals()[rnncell_type]

        if self.num_rnns:
            rnn_cells = []
            for i in range(self.num_rnns):
                rnn_cells.append(rnncell_obj(**rnn_cfgs[i]))
            self.rnn_cells = nn.ModuleList(rnn_cells)

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
        if self.num_rnns > 0:
            for rnn_cell in self.rnn_cells:
                rnn_cell.init_weights()
        if self.num_extra_convs > 0:
            for conv in self.extra_convs:
                xavier_init(conv)

    def forward(self, input_list, in_dict=None, is_train=False):
        # First process time dimension in lower layer, then go up
        # This will affect how state_dict updates.
        time, batch, chan, _, _ = input_list[0].shape

        if is_train or in_dict is None:
            device = input_list[0].device
            states_list = []
            rnn_lvl = 0
            for lvl, tp in enumerate(self.out_layers_type):
                if tp == 1:
                    _t, _b, _c, _h, _w = input_list[lvl].shape
                    in_shape = [_b, _c, _h, _w]
                    states_list.append(
                        self.rnn_cells[rnn_lvl].init_state(in_shape, device))
                    rnn_lvl += 1
            in_dict = dict(states_list=states_list)

        states_list = in_dict['states_list']

        final_outs = []
        out_dict_lists = []
        rnn_level = 0
        extra_level = 0
        feats = None
        for lvl, inputs in enumerate(input_list):
            if self.out_layers_type[lvl] == 0:
                # Use exists
                out = inputs.view([-1, *inputs.shape[2:]])

            elif self.out_layers_type[lvl] == 1:
                # Use output of LSTM
                feats, state_dict = self.forward_rnn(
                    inputs,
                    self.rnn_cells[rnn_level],
                    states=states_list[rnn_level])
                feats = feats.view([-1, *feats.shape[2:]])
                out = feats
                rnn_level += 1
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
    def forward_rnn(inputs, rnn_cell, states):
        decoder_outs = []
        for decoder_input in inputs:
            feat, states = rnn_cell(decoder_input, states)
            decoder_outs.append(feat)
        decoder_outs = torch.stack(decoder_outs, dim=0)
        return decoder_outs, states
