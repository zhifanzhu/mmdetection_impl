import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init
from mmdet.ops import MxCorrelation, MxAssemble

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class CA(nn.Module):

    def __init__(self,
                 in_channels,
                 corr_disp,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 conv_final=True):
        super(CA, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']
        self.corr = MxCorrelation(pad_size=corr_disp, kernel_size=1,
                                  max_displacement=corr_disp, stride1=1, stride2=1)
        self.assemble = MxAssemble(k=corr_disp)

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)

        if conv_final:
            self.conv_final = nn.Sequential(
                ConvModule(
                    in_channels=2*in_channels,
                    out_channels=256,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    activation='relu'),
                ConvModule(
                    in_channels=256,
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    activation='relu'),
                ConvModule(
                    in_channels=16,
                    out_channels=3,
                    kernel_size=3,
                    padding=1,
                    stride=1),
                nn.Conv2d(
                    in_channels=3,
                    out_channels=1,
                    kernel_size=3,
                    padding=1,
                    stride=1)
            )

        if self.conv_final is not None:
            self.init_weights(zeros_init=False)
        else:
            self.init_weights(zeros_init=True)

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)
        if self.conv_final is not None:
            nn.init.constant_(self.conv_final[-1].bias, 0.0)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, DxD, H, W]
        pairwise_weight = self.corr(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight *= 256  # sumelems = kernel*kernel* bottomchannels
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, DxD, H, W]
        pairwise_weight = self.corr(theta_x, phi_x)
        pairwise_weight *= 256
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x, x_ref):
        """ g is ref value, phi is ref key, theta is current(query)"""
        n, _, h, w = x.shape

        g_x = self.g(x_ref)
        theta_x = self.theta(x)
        phi_x = self.phi(x_ref)

        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)

        y = self.assemble(pairwise_weight, g_x)

        if self.conv_final is not None:
            y = self.conv_out(y)
            cat_feat = torch.cat([x, y], dim=1)
            output = x + self.conv_final(cat_feat) * y
        else:
            output = x + self.conv_out(y)

        return output


@PAIR_MODULE.register_module
class TwinEmbedCA(nn.Module):

    def __init__(self, channels=256, reduction=2, conv_final=True, against_self=False):
        super(TwinEmbedCA, self).__init__()
        kwargs = dict(
            use_scale=True,
            mode='embedded_gaussian',
            conv_final=conv_final)
        self.top_conv = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            stride=2)
        self.blocks = nn.ModuleList([
            CA(
                in_channels=channels,
                corr_disp=1,
                reduction=reduction,
                **kwargs),
            CA(
                in_channels=channels,
                corr_disp=1,
                reduction=reduction,
                **kwargs),
            CA(
                in_channels=channels,
                corr_disp=1,
                reduction=reduction,
                **kwargs),
            CA(
                in_channels=channels,
                corr_disp=1,
                reduction=reduction,
                **kwargs),
            CA(
                in_channels=channels,
                corr_disp=1,
                reduction=reduction,
                **kwargs),
        ])
        self.against_self = against_self

    def init_weights(self):
        for g in self.blocks:
            g.init_weights()
        self.top_conv.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        if self.against_self:
            outs = [
                self.blocks[0](x=feat[0], x_ref=feat[0]),
                self.blocks[1](x=feat[1], x_ref=feat[1]),
                self.blocks[2](x=feat[2], x_ref=feat[2]),
                self.blocks[3](x=feat[3], x_ref=feat[3]),
                self.blocks[4](x=feat[4], x_ref=feat[4]),
            ]
        else:
            outs = [
                self.blocks[0](x=feat[0], x_ref=feat_ref[1]),
                self.blocks[1](x=feat[1], x_ref=feat_ref[2]),
                self.blocks[2](x=feat[2], x_ref=feat_ref[3]),
                self.blocks[3](x=feat[3], x_ref=feat_ref[4]),
                self.blocks[4](x=feat[4], x_ref=self.top_conv(feat_ref[4])),
            ]
        return outs
