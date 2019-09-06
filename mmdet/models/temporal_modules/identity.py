import torch
import torch.nn as nn

from ..registry import TEMPORAL_MODULE
from mmdet.core import multi_apply


"""
Temporal module takes care of transforming input data its own favor,
e.g. Aggregate temporal info using add, concat, LSTM, etc.
And transform feature back for output.

self.forward(x_seq) receives input like:
x_seq: a list(repr multilevel feature, typical 5) of [T, B, C, H, W] Tensor.

and is responsible for transforming output to bbox_head compatible format,
i.e. [[B*T, C, H, W]*5] (Or [[B*T, C, H, W]*6] like LSTM-SSD).


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

    def forward(self, inputs_list):
        return tuple([
            self.forward_single(v) for v in inputs_list
        ])

    def forward_single(self, inputs):
        time, batch, c, h, w = inputs.shape

        # Do a dummy rnn step
        outputs = []
        for i, decoder_input in enumerate(inputs):
            out = self.decoder(decoder_input)
            outputs.append(out)
        outputs = torch.stack(outputs)
        # Transform back
        outputs = outputs.permute([1, 0, 2, 3, 4]).reshape([batch*time, c, h, w])
        return outputs
