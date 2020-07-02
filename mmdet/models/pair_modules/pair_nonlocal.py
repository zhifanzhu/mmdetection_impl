import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

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
                 mode='embedded_gaussian',
                 conv_final=False):
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

    def forward(self, x, x_ref, pairwise_weight=None):
        """ g is ref value, phi is ref key, theta is current(query)"""
        n, _, h, w = x.shape

        # g_x: [N, HxW, C]
        g_x = self.g(x_ref).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if pairwise_weight is None:
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

        if hasattr(self, 'conv_final'):
            y = self.conv_out(y)
            cat_feat = torch.cat([x, y], dim=1)
            output = x + self.conv_final(cat_feat) * y
        else:
            output = x + self.conv_out(y)

        return output, pairwise_weight


@PAIR_MODULE.register_module
class PairNonLocal(nn.Module):

    def __init__(self, channels=256, reduction=2, conv_final=False):
        super(PairNonLocal, self).__init__()
        self.nl_blocks = nn.ModuleList([
            NonLocal2D(
                in_channels=channels,
                reduction=reduction,
                use_scale=True,
                mode='embedded_gaussian',
                conv_final=conv_final)
            for _ in range(5)])

    def init_weights(self):
        for g in self.nl_blocks:
            g.init_weights()

    def forward(self, feat, feat_ref, is_train=False):
        outs = [
            self.nl_blocks[0](x=feat[0], x_ref=feat_ref[0])[0],
            self.nl_blocks[1](x=feat[1], x_ref=feat_ref[1])[0],
            self.nl_blocks[2](x=feat[2], x_ref=feat_ref[2])[0],
            self.nl_blocks[3](x=feat[3], x_ref=feat_ref[3])[0],
            self.nl_blocks[4](x=feat[4], x_ref=feat_ref[4])[0],
        ]
        return outs


@PAIR_MODULE.register_module
class PairReuseWeight(nn.Module):

    def __init__(self, channels=256, reduction=2, conv_final=False):
        super(PairReuseWeight, self).__init__()
        self.nl_blocks = nn.ModuleList([
            NonLocal2D(
                in_channels=channels,
                reduction=reduction,
                use_scale=True,
                mode='embedded_gaussian',
                conv_final=conv_final)
            for _ in range(5)])

    def init_weights(self):
        for g in self.nl_blocks:
            g.init_weights()

    def forward(self, feat, feat_ref, pw_list=None, is_train=False):
        if pw_list is None:
            pw_list = [None for _ in range(len(feat))]
        outs, pws = [], []
        for i in range(len(feat)):
            out, pw = self.nl_blocks[i](feat[i], feat_ref[i], pw_list[i])
            outs.append(out)
            pws.append(pw)
        return outs, pws
