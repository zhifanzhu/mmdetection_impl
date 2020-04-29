import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init
from mmdet.ops import MxCorrelation, MxAssemble

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class NonLocal2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

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

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x, x_ref):
        """ g is ref value, phi is ref key, theta is current(query)"""
        n, _, h, w = x.shape

        # g_x: [N, HxW, C]
        g_x = self.g(x_ref).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, HxW]
        phi_x = self.phi(x_ref).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        return output


class CA(nn.Module):

    def __init__(self,
                 in_channels,
                 corr_disp,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
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

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

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

        output = x + self.conv_out(y)

        return output


@PAIR_MODULE.register_module
class EmbedCA(nn.Module):

    def __init__(self, channels=256, reduction=2):
        super(EmbedCA, self).__init__()
        kwargs = dict(use_scale=True, mode='embedded_gaussian')
        self.blocks = nn.ModuleList([
            CA(
                in_channels=channels,
                corr_disp=6,
                reduction=reduction,
                **kwargs),
            CA(
                in_channels=channels,
                corr_disp=4,
                reduction=reduction,
                **kwargs),
            CA(
                in_channels=channels,
                corr_disp=2,
                reduction=reduction,
                **kwargs),
            NonLocal2D(
                in_channels=channels,
                reduction=reduction,
                **kwargs),
            NonLocal2D(
                in_channels=channels,
                reduction=reduction,
                **kwargs),
        ])

    def init_weights(self):
        for g in self.blocks:
            g.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        outs = [
            self.blocks[0](x=feat[0], x_ref=feat_ref[0]),
            self.blocks[1](x=feat[1], x_ref=feat_ref[1]),
            self.blocks[2](x=feat[2], x_ref=feat_ref[2]),
            self.blocks[0](x=feat[3], x_ref=feat_ref[3]),
            self.blocks[1](x=feat[4], x_ref=feat_ref[4]),
        ]
        return outs
