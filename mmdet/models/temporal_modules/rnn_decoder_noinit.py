import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import temporal_modules

from mmcv.cnn import xavier_init

from ..registry import TEMPORAL_MODULE
"""
This class is VERY similar to RNNDecoder, except that it does not init state for 
first frame.
"""


@TEMPORAL_MODULE.register_module
class NoInitRNNDecoder(nn.Module):
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
        super(NoInitRNNDecoder, self).__init__()
        assert len(in_channels) == len(out_layers_type)
        self.neck_first = neck_first
        self.out_layers_type = out_layers_type
        self.num_olds = sum([1 for t in out_layers_type if t == 0])
        self.num_rnns = sum([1 for t in out_layers_type if t == 1])
        self.num_extra_convs = sum([1 for t in out_layers_type if t == 2])
        assert len(rnn_cfgs) == self.num_rnns
        rnncell_obj = getattr(temporal_modules, rnncell_type)

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
                        in_channels=in_channels[l-1],
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
            states_list = []
            rnn_lvl = 0
            for lvl, tp in enumerate(self.out_layers_type):
                if tp == 1:
                    states_list.append(None)  # None is a placeholder for check in forward_rnn()
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
        if states is not None:
            for decoder_input in inputs:
                feat, states = rnn_cell(decoder_input, states)
                decoder_outs.append(feat)
            decoder_outs = torch.stack(decoder_outs, dim=0)
        else:
            states = inputs[0, ...]
            decoder_outs.append(states)  # append first frame feat directly
            for decoder_input in inputs[1:, ...]:
                feat, states = rnn_cell(decoder_input, states)
                decoder_outs.append(feat)
            decoder_outs = torch.stack(decoder_outs, dim=0)
        return decoder_outs, states
