import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn import constant_init, normal_init

from ..registry import PAIR_MODULE
from ..utils import ConvModule


class Attention(nn.Module):
    """
    `g` output 256, not 128, which differs from Nonlocal

    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(Attention, self).__init__()
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
            self.in_channels,
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
        g_x = self.g(x_ref).view(n, self.in_channels, -1)
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
        y = y.permute(0, 2, 1).reshape(n, self.in_channels, h, w)

        return y


class ThreeWay(nn.Module):

    def __init__(self, channels=256):
        """

        :param channels:
        """
        super(ThreeWay, self).__init__()
        final_chan = 1
        chans = [256, 16, 3, final_chan]
        self.conv_final = nn.Sequential(
            ConvModule(
                in_channels=2*channels,
                out_channels=chans[0],
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[0],
                out_channels=chans[1],
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[1],
                out_channels=chans[2],
                kernel_size=3,
                padding=1,
                stride=1),
            nn.Conv2d(
                in_channels=chans[2],
                out_channels=chans[3],
                kernel_size=3,
                padding=1,
                stride=1))
        self.atten = Attention(channels)
        self.atten_conv_final = nn.Sequential(
            ConvModule(
                in_channels=2*channels,
                out_channels=chans[0],
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[0],
                out_channels=chans[1],
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[1],
                out_channels=chans[2],
                kernel_size=3,
                padding=1,
                stride=1),
            nn.Conv2d(
                in_channels=chans[2],
                out_channels=chans[3],
                kernel_size=3,
                padding=1,
                stride=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        nn.init.constant_(self.conv_final[-1].bias, 0.0)
        nn.init.constant_(self.atten_conv_final[-1].bias, 0.0)
        self.atten.init_weights()

    def forward_base(self, f, f_h):
        cat_feat = torch.cat([f, f_h], dim=1)
        way_1 = self.conv_final(cat_feat) * f_h

        attened = self.atten(f, f_h)
        way_2 = self.atten_conv_final(
            torch.cat([f, attened], dim=1)) * attened

        return way_1, way_2

    def forward(self, f, f_h):
        way_1, way_2 = self.forward_base(f, f_h)
        return f + way_1 + way_2


class ThreeWayPixelSelect(nn.Module):

    def __init__(self, channels=256):
        super(ThreeWayPixelSelect, self).__init__()
        self.base = ThreeWay(channels)

        self.channels = channels
        chans = [256, 16, 3, 2]
        self.select_net = nn.Sequential(
            ConvModule(
                in_channels=2*self.channels,
                out_channels=chans[0],
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[0],
                out_channels=chans[1],
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=chans[1],
                out_channels=chans[2],
                kernel_size=3,
                padding=1,
                stride=1),
            nn.Conv2d(
                in_channels=chans[2],
                out_channels=chans[3],
                kernel_size=3,
                padding=1,
                stride=1))

    def init_weights(self):
        self.base.init_weights()
        nn.init.constant_(self.select_net[-1].bias[0], 0.5)
        nn.init.constant_(self.select_net[-1].bias[1], 0.5)

    def forward(self, f, f_h):
        way_1, way_2 = self.base.forward_base(f, f_h)
        score = torch.softmax(torch.cat([way_1, way_2], dim=1), dim=1)
        way_merge = score[:, 0, :, :].unsqueeze(1) * way_1 + \
                    score[:, 1, :, :].unsqueeze(1) * way_2

        return f + way_merge


class ThreeWayChannelSelect(nn.Module):

    def __init__(self, channels=256):
        super(ThreeWayChannelSelect, self).__init__()
        self.base = ThreeWay(channels)
        self.channels = channels
        self.select_net = nn.Sequential(
            ConvModule(
                in_channels=2*self.channels,
                out_channels=self.channels,
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                stride=1,
                activation='relu'),
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=1,
                padding=0,
                stride=1,
                activation='relu'),
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=1,
                padding=0,
                stride=1),
            nn.Sigmoid())

    def init_weights(self):
        self.base.init_weights()

    def forward(self, f, f_h):
        way_1, way_2 = self.base.forward_base(f, f_h)
        score = self.select_net(torch.cat([way_1, way_2], dim=1))
        way_merge = score * way_1 + (1 - score) * way_2
        return f + way_merge


@PAIR_MODULE.register_module
class TwinThreeWay(nn.Module):

    def __init__(self,
                 channels=256,
                 select='none'):
        """

        Args:
            channels:
            select:  'none', 'pixel', 'channel'
        """
        super(TwinThreeWay, self).__init__()
        if select == 'none':
            self.grab = ThreeWay(channels)
        elif select == 'pixel':
            self.grab = ThreeWayPixelSelect(channels)
        elif select == 'channel':
            self.grab = ThreeWayChannelSelect(channels)
        else:
            raise ValueError('select not understood.')

        self.conv_extra = ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=2,           # Note different stride
                activation='relu')

    def init_weights(self):
        self.grab.init_weights()
        for m in self.conv_extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feat, feat_ref, is_train=False):
        outs = [
            self.grab(f=feat[0], f_h=feat_ref[1]),
            self.grab(f=feat[1], f_h=feat_ref[2]),
            self.grab(f=feat[2], f_h=feat_ref[3]),
            self.grab(f=feat[3], f_h=feat_ref[4]),
        ]
        feat_ref_top = self.conv_extra(feat_ref[4])
        outs.append(self.grab(f=feat[4], f_h=feat_ref_top))

        return outs
