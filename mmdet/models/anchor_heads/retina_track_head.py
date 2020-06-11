import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead


@HEADS.register_module
class RetinaTrackHead(AnchorHead):
    """
    An anchor-based head used in [1]_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 m1,
                 m2,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 freeze_all=False,
                 version='v1',
                 **kwargs):
        self.m1 = m1
        self.m2 = m2
        self.version = version
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.freeze_all = freeze_all
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaTrackHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)

        # Task shared
        self.m1_convs = nn.ModuleList()
        if self.version == 'v1':
            for k in range(self.num_anchors):
                m1_branch = nn.ModuleList()
                for i in range(self.m1):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    m1_branch.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg))
                self.m1_convs.append(m1_branch)
        elif self.version =='v2':
            for i in range(self.m1):
                chn = self.in_channels if i == 0 else self.feat_channels
                chn = chn * self.num_anchors
                self.m1_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels * self.num_anchors,
                        3,
                        stride=1,
                        padding=1,
                        groups=self.num_anchors,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.m2):
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)

        if self.freeze_all:
            def _freeze_conv(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    m.requires_grad = False
            self.apply(_freeze_conv)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        num_img, _, height, width = x.shape
        if self.version == 'v1':
            task_shared_outs = []
            for k in range(self.num_anchors):
                feat = x
                for i in range(self.m1):
                    feat = self.m1_convs[k][i](feat)
                task_shared_outs.append(feat)
            task_shared_feat = torch.cat(task_shared_outs, 0)
        else:
            feat = x.repeat(1, self.num_anchors, 1, 1)
            for i in range(self.m1):
                feat = self.m1_convs[i](feat)
            task_shared_feat = feat

        cls_feat = task_shared_feat
        reg_feat = task_shared_feat
        for i in range(self.m2):
            cls_feat = self.cls_convs[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_feat)

        cls_score = self.retina_cls(cls_feat).view(
            num_img, self.cls_out_channels * self.num_anchors, height, width)
        bbox_pred = self.retina_reg(reg_feat).view(
            num_img, 4 * self.num_anchors, height, width)
        return cls_score, bbox_pred
