import torch
import torch.nn as nn

from ..registry import TEMPORAL_MODULE


"""
Temporal module takes care of transforming input data its own favor,
e.g. Aggregate temporal info using add, concat, LSTM, etc.
And transform feature back for output.

self.forward_train(x_seq) receives input like:
x_seq: a list(repr multilevel feature, typical 5) of [T, B, C, H, W] Tensor.

and is responsible for transforming output to bbox_head compatible format,
i.e. [[B*T, C, H, W]*5] (Or [[B*T, C, H, W]*6] like LSTM-SSD).

During test, input shape:
x: [1*1, C, H, W] * 5]


"""


@TEMPORAL_MODULE.register_module
class Identity(nn.Module):
    """ Identity temporal module, i.e. no modification on input data.
    """

    def __init__(self):
        super(Identity, self).__init__()
        self.decoder = nn.Sequential()  # Identity Module

    def init_weights(self):
        pass

    def forward(self, inputs_list, in_dict=None, is_train=False):
        out_feats = []
        out_states = []
        device = inputs_list[0].device
        if is_train:
            in_states = [torch.zeros(*v.shape[1:]).to(device)
                         for v in inputs_list]
            # [B, C, H, W] * 4
            for inputs, in_state in zip(inputs_list, in_states):
                out_feat, out_state = self.forward_single(inputs, in_state)
                out_feats.append(out_feat)
                out_states.append(out_state)
            out_dict = dict(states=out_states)
            return tuple(out_feats), out_dict

        # Test phase
        if in_dict is None or in_dict.get('states', None) is None:
            in_states = [torch.zeros(*v.shape).to(device)
                         for v in inputs_list]
        else:
            in_states = in_dict['states']
        for inputs, in_state in zip(inputs_list, in_states):
            out_feat, out_state = self.forward_single(inputs, in_state)
            out_feats.append(out_feat)
            out_states.append(out_state)
        out_dict = dict(states=out_states)
        return tuple(out_feats), out_dict

    def forward_single(self, inputs, in_state):
        time, batch, c, h, w = inputs.shape

        # Do a dummy rnn step
        outputs = []
        out_state = in_state
        for i, decoder_input in enumerate(inputs):
            out = self.decoder(decoder_input) + out_state
            out_state = out_state  # Dummy update
            outputs.append(out)
        outputs = torch.stack(outputs)
        # Transform back
        outputs = outputs.permute([1, 0, 2, 3, 4]).reshape([batch*time, c, h, w])
        return outputs, out_state
