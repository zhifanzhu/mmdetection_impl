import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init, xavier_init

from ..registry import HEADS
from ..utils import ConvModule, ConvModuleLite, bias_init_with_prob
from .anchor_head import AnchorHead


@HEADS.register_module
class RetinaHead(AnchorHead):
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
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 freeze_all=False,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.freeze_all = freeze_all
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

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
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred


@HEADS.register_module
class RetinaHeadLite(RetinaHead):
    """ See RetinaHead and MobileNetV2.

        init_method: 'normal' or 'xavier' (tensorflow)
    """

    def __init__(self,
                 dummy=False,
                 separable_final='false',
                 init_method='normal',
                 **kwargs):
        self.dummy = dummy
        self.separable_final = separable_final
        self.init_method = init_method
        super(RetinaHeadLite, self).__init__(**kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU6(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        if self.dummy:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
        else:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModuleLite(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        activation='relu',
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModuleLite(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        activation='relu',
                        norm_cfg=self.norm_cfg))
        if self.separable_final:
            self.retina_cls = nn.Sequential(
                # dw
                nn.Conv2d(self.feat_channels,
                          self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=self.feat_channels,
                          bias=False),
                # pw-linear
                nn.Conv2d(self.feat_channels,
                          self.num_anchors * self.cls_out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True))
            self.retina_reg = nn.Sequential(
                # dw
                nn.Conv2d(self.feat_channels,
                          self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          groups=self.feat_channels,
                          bias=False),
                # pw-linear
                nn.Conv2d(self.feat_channels,
                          self.num_anchors * 4,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True))
        else:
            self.retina_cls = nn.Conv2d(
                self.feat_channels,
                self.num_anchors * self.cls_out_channels,
                3,
                padding=1)
            self.retina_reg = nn.Conv2d(
                self.feat_channels, self.num_anchors * 4, 3, padding=1)

        if self.freeze_all:
            def _freeze_conv(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    m.requires_grad = False
            self.apply(_freeze_conv)

    def init_weights(self):
        if self.dummy:
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.reg_convs:
                normal_init(m.conv, std=0.01)
        else:
            def _init_conv_normal(m):
                classname = m.__class__.__name__
                # Make sure it finds real nn.Conv2d
                if classname.find('Conv') != -1 and \
                        classname != 'ConvModuleLite':
                    normal_init(m, std=0.01)

            def _init_conv_xavier(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1 and \
                        classname != 'ConvModuleLite':
                    xavier_init(m)

            if self.init_method == 'normal':
                self.cls_convs.apply(_init_conv_normal)
                self.reg_convs.apply(_init_conv_normal)
            elif self.init_method == 'xavier':
                self.cls_convs.apply(_init_conv_xavier)
                self.reg_convs.apply(_init_conv_xavier)
            else:
                raise ValueError(
                    f"Unsupported init_method: {self.init_method}")
        bias_cls = bias_init_with_prob(0.01)
        if self.separable_final:
            normal_init(self.retina_cls[0], std=0.01)
            normal_init(self.retina_reg[0], std=0.01)
            normal_init(self.retina_cls[1], std=0.01, bias=bias_cls)
            normal_init(self.retina_reg[1], std=0.01)
        else:
            normal_init(self.retina_cls, std=0.01, bias=bias_cls)
            normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
