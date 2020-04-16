import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import Correlation, FastAssemble, MxCorrelation

from ..registry import PAIR_MODULE

"""
Pyramid correlation.
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
        nn.init.constant_(self.conv[-1].bias[0], 1.0)
        nn.init.constant_(self.conv[-1].bias[1], 0.0)

    def forward(self, feat, aligned_ref):
        cat = torch.cat([feat, aligned_ref], dim=1)
        conv = self.conv(cat)
        score = torch.softmax(conv, dim=1)
        out = score[:, 0, :, :].unsqueeze(1) * feat + \
                score[:, 1, :, :].unsqueeze(1) * aligned_ref
        return out


@PAIR_MODULE.register_module
class PyCorrAssemble(nn.Module):
    """
    Pyramid correlation processing
    feat0: [torch.Size([1, 256, 64, 64]),
    feat1: torch.Size([1, 256, 32, 32]),
    feat2: torch.Size([1, 256, 16, 16]),
    feat3: torch.Size([1, 256, 8, 8]),
    feat4: torch.Size([1, 256, 4, 4])]
    """

    def __init__(self,
                 alpha=0.5,
                 beta=0.5,
                 mode='bilinear',
                 neck_first=True,
                 use_mxcorr=True,
                 ):
        super(PyCorrAssemble, self).__init__()
        if use_mxcorr:
            self.corr = MxCorrelation(pad_size=1, kernel_size=1, max_displacement=1,
                                      stride1=1, stride2=1)
        else:
            self.corr = Correlation(pad_size=1, kernel_size=1, max_displacement=1,
                                    stride1=1, stride2=1)

        self.assem0 = FastAssemble(k=4)
        self.assem1 = FastAssemble(k=2)
        self.assem2 = FastAssemble(k=1)

        self.upd_net0 = ConcatUpdate(in_channels=256)
        self.upd_net1 = ConcatUpdate(in_channels=256)
        self.upd_net2 = ConcatUpdate(in_channels=256)

        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.neck_first = neck_first

    def mix_corr(self, corr_bot, corr_top):
        """
        upsample corr_top, add corr_bot into upsampled.
        corr_top and corr_bot should have same shape.
        e.g.
            mix_corr([b, 9, 32, 32], [b, 9, 16, 16]) -> [b, 25, 32, 32]
            mix_corr([b, 9, 64, 64], [b, 25, 32, 32]) -> [b, 81, 64, 64]
            corr_update = alpha * corr_bot + beta * corr_top
        """
        MAP = {9: 3, 25: 5, 91: 9}
        nb, c_top, h, w = corr_top.shape
        _, c_bot, h2, w2 = corr_bot.shape  # h2, w2 = 2*h, 2*w
        d_bot = MAP[c_bot]
        d_top = MAP[c_top]
        d_top2 = 2 * d_top - 1  # 2x(2x (d-1) / 2) + 1
        size = (h2, w2)
        corr = F.interpolate(corr_top, size=size, mode=self.mode, align_corners=None)
        corr = corr.permute(0, 2, 3, 1).view(nb, h2*w2, d_top, d_top)
        corr_exp = F.interpolate(corr, size=(d_top2, d_top2), mode=self.mode, align_corners=None)
        corr_exp = self.beta * corr_exp
        i_bgn = (d_top2 - d_bot) // 2
        i_end = i_bgn + d_bot
        corr_exp[:, :, i_bgn:i_end, i_bgn:i_end] = \
            corr_exp[:, :, i_bgn:i_end, i_bgn:i_end] + \
            self.alpha * corr_bot.view(nb, h2*w2, d_bot, d_bot)
        return corr_exp.view(nb, h2, w2, d_top2*d_top2).permute(0, 3, 1, 2)

    def init_weights(self):
        self.upd_net0.init_weights()
        self.upd_net1.init_weights()
        self.upd_net2.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        feat0, feat1, feat2, feat3, feat4 = feat
        feat0_ref, feat1_ref, feat2_ref, feat3_ref, feat4_ref = feat_ref
        # aff3 = self.corr(feat3, feat3_ref)
        aff2 = self.corr(feat2, feat2_ref)
        aff1 = self.corr(feat1, feat1_ref)
        aff0 = self.corr(feat0, feat0_ref)

        aff1 = self.mix_corr(aff1, aff2)  # [b, 25, 16, 16]
        aff0 = self.mix_corr(aff0, aff1)  # [b, 81, 32, 32]
        if self.use_softmax_norm:
            aff0 = torch.softmax(aff0, dim=1)
            aff1 = torch.softmax(aff1, dim=1)
            aff2 = torch.softmax(aff2, dim=1)
        else:
            aff0 = aff0 / (torch.sum(aff0, dim=1, keepdim=True) + 1e-7)
            aff1 = aff1 / (torch.sum(aff1, dim=1, keepdim=True) + 1e-7)
            aff2 = aff2 / (torch.sum(aff2, dim=1, keepdim=True) + 1e-7)
        feat0_upd = self.assem0(aff0, feat0_ref)
        feat1_upd = self.assem1(aff1, feat1_ref)
        feat2_upd = self.assem2(aff2, feat2)
        updated_cur_feat = [
            self.upd_net0(feat0, feat0_upd),
            self.upd_net1(feat1, feat1_upd),
            self.upd_net2(feat2, feat2_upd),
            feat3,
            feat4,
        ]
        return updated_cur_feat
