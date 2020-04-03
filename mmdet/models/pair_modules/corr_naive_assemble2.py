import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.ops import Correlation, NaiveAssemble2

from ..registry import PAIR_MODULE

"""
Simplified from TEMPORAL_MODULE.
This file is a demonstration.
"""


class ConcatUpdate(nn.Module):
    """ Update Net like """
    def __init__(self,
                 in_channels):
        super(ConcatUpdate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=3,
                out_channels=2,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )

    def init_weights(self):
        def _init_conv(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                normal_init(m, std=0.01)
        self.apply(_init_conv)
        nn.init.const_(self.conv[-1].bias[0], 1.0)
        nn.init.const_(self.conv[-1].bias[1], 0.0)

    def forward(self, feat, aligned_ref):
        cat = torch.cat([feat, aligned_ref], dim=1)
        conv = self.conv(cat)
        score = torch.softmax(conv, dim=1)
        out = score[:, 0, :, :].unsqueeze(1) * feat + \
                score[:, 1, :, :].unsqueeze(1) * aligned_ref
        return out


class ConcatSkip(nn.Module):
    """ Update Net like """
    def __init__(self,
                 in_channels):
        super(ConcatSkip, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=3,
                out_channels=1,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )

    def init_weights(self):
        def _init_conv(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                normal_init(m, std=0.01)
        self.apply(_init_conv)
        nn.init.const_(self.conv[-1].bias, 0.0)

    def forward(self, feat, aligned_ref):
        cat = torch.cat([feat, aligned_ref], dim=1)
        score = self.conv(cat)
        out = feat + score * aligned_ref
        return out


class RFU(nn.Module):
    def __init__(self,
                 corr_disp,
                 in_channels,
                 use_softmax_norm=False,
                 use_add=False,
                 use_concat_skip=False,
                 ):
        super(RFU, self).__init__()
        self.corr = Correlation(corr_disp, kernel_size=1,
                                max_displacement=corr_disp, stride1=1, stride2=1)
        self.assemble = NaiveAssemble2(
            pad_size=corr_disp, kernel_size=1,
            max_displacement=corr_disp, stride1=1, stride2=1)
        if use_concat_skip:
            self.update_net = ConcatSkip(in_channels)
        else:
            self.update_net = ConcatUpdate(in_channels)

        self.use_softmax_norm = use_softmax_norm
        self.use_add = use_add

    def init_weights(self):
        self.update_net.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        if not self.use_add:
            aff = self.corr(feat, feat_ref)
            if self.use_softmax_norm:
                aff = torch.softmax(aff, dim=1)
            else:
                aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-7)
            aligned_ref = self.assemble(aff, feat_ref)
        else:
            aligned_ref = feat + feat_ref
        updated_cur_feat = self.update_net(feat, aligned_ref)
        return updated_cur_feat


@PAIR_MODULE.register_module
class CorrNaiveAssemble2(nn.Module):
    """
    [torch.Size([1, 256, 64, 64]),  <-- only process this for now.
     torch.Size([1, 256, 32, 32]),
     torch.Size([1, 256, 16, 16]),
     torch.Size([1, 256, 8, 8]),
     torch.Size([1, 256, 4, 4])]
    """

    def __init__(self,
                 disp,
                 neck_first,

                 layers=(0,),
                 use_softmax_norm=False,
                 use_add=False,
                 use_concat_skip=False,
                 ):
        super(CorrNaiveAssemble2, self).__init__()
        self.rfu_64 = RFU(disp, 256, use_softmax_norm, use_add, use_concat_skip)
        self.neck_first = neck_first
        self.trans_layers = [True if l in layers else False for l in range(5)]

    def init_weights(self):
        self.rfu_64.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        out = [
            self.rfu_64(feat[l], feat_ref[l], is_train)
            if t else feat[l]
            for l, t in enumerate(self.trans_layers)
        ]
        return out


@PAIR_MODULE.register_module
class MultiCorrNaiveAssemble2(nn.Module):
    """
    Stand Alone rfu for each layer
    """

    def __init__(self,
                 disp,
                 neck_first,

                 layers=(0,),
                 use_add=False,
                 ):
        super(MultiCorrNaiveAssemble2, self).__init__()
        self.neck_first = neck_first

        self.rfu_list = nn.ModuleList()
        for _ in layers:
            self.rfu_list.append(RFU(disp, 256, False, use_add, False))
        self.trans_layers = [True if l in layers else False for l in range(5)]

    def init_weights(self):
        for rfu in self.rfu_list:
            rfu.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        out = [
            self.rfu_list[l](feat[l], feat_ref[l], is_train)
            if t else feat[l]
            for l, t in enumerate(self.trans_layers)
        ]
        return out
