import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.ops import DeformConv, ModulatedDeformConv

from ..registry import TEMPORAL_MODULE

"""
Given input of (128, 64, 32), first calculate corr on them,
then on cat to make 32x32 feat map,
then conv to produce 16x16,
then use it as offset field to dconv in_channels[adapt_layer].
Perform corr before neck. 
"""


@TEMPORAL_MODULE.register_module
class CorrelationAdaptor(nn.Module):

    def __init__(self,
                 in_channels=(256, 512, 1024, 2048),
                 adapt_layer=3,
                 # out_adapt_channel=256,
                 displacements=(32, 16, 8),
                 strides=(4, 2, 1),
                 kernel_size=3,
                 deformable_groups=4,
                 neck_first=False,
                 with_modulated_dcn=False):
        """

        Args:
            in_channels: [256, 512, 1024, 2048] for retina res50.
            displacements:
            strides:
            kernel_size:
            deformable_groups:
            neck_first:
        """

        super(CorrelationAdaptor, self).__init__()
        raise NotImplementedError
        self.adapt_layer = adapt_layer
        self.neck_first = neck_first
        self.with_modulated_dcn = with_modulated_dcn

        assert len(displacements) == len(strides)
        self.corrs = nn.ModuleList([
            Correlation(pad_size=disp, kernel_size=1, max_displacement=disp,
                        stride1=s, stride2=s)
            for disp, s in zip(displacements, strides)
        ])
        if self.with_modulated_dcn:
            offset_channels = kernel_size * kernel_size * 3
            conv_op = ModulatedDeformConv
        else:
            offset_channels = kernel_size * kernel_size * 2
            conv_op = DeformConv

        # num_corr_channels = 867 , too high dim ?
        num_corr_channels = int(sum([(2*(d/s) + 1)**2
                                     for d, s in zip(displacements, strides)]))
        bottleneck = 256
        self.corr_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_corr_channels,
                out_channels=bottleneck,
                kernel_size=3,
                padding=(kernel_size - 1) // 2,
                stride=2),
            nn.ReLU(inplace=True))

        self.conv_offset = nn.Conv2d(
            bottleneck,
            deformable_groups * offset_channels,
            kernel_size=1,
            bias=False)
        self.conv_adaption = conv_op(
            in_channels[adapt_layer],
            in_channels[adapt_layer],
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):

        def _init_conv(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                normal_init(m, std=0.01)
        self.apply(_init_conv)

    # TODO(zhifan) use previous in_dict during test.
    def forward(self, input_list, in_dict=None, is_train=False):
        time, batch, _, _, _ = input_list[0].shape
        outs = []
        for l, inputs in enumerate(input_list[:self.adapt_layer]):
            outs.append(inputs.view(time*batch,
                                    inputs.size(2),
                                    inputs.size(3),
                                    inputs.size(4)))

        if time > 1:
            corr_c2 = self.forward_corr(input_list[0], level=0)
            corr_c3 = self.forward_corr(input_list[1], level=1)
            corr_c4 = self.forward_corr(input_list[2], level=2)
            corr_cat = torch.cat([corr_c2, corr_c3, corr_c4], dim=1)  # [(T-1)*B, c_new, h ,w]
            corr_feat = self.corr_conv(corr_cat)
            feat_c5 = input_list[3]
            feat_c5_input = feat_c5[1:].view(
                (time - 1) * batch, feat_c5.size(2), feat_c5.size(3), feat_c5.size(4))
            if self.with_modulated_dcn:
                offset_mask = self.conv_offset(corr_feat)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                feat_adapt = self.relu(
                    self.conv_adaption(feat_c5_input, offset, mask))
            else:
                offset = self.conv_offset(corr_feat)
                feat_adapt = self.relu(self.conv_adaption(feat_c5_input, offset))
            feat_adapt = torch.cat([feat_c5[0], feat_adapt], dim=0)
        else:
            feat_c5 = input_list[3]
            feat_adapt = input_list[3].view(
                batch, feat_c5.size(2), feat_c5.size(3), feat_c5.size(4))
        outs.append(feat_adapt)

        if is_train:
            return outs, None
        else:
            # return inputs for reuse in next frame/clips
            out_dict = dict(
                inputs=input_list
            )
            return outs, out_dict

    def forward_corr(self, inputs, level):
        time, batch, c, h, w = inputs.shape
        if time <= 1:
            return inputs.view(-1, c, h, w)

        corr_feats = []
        for t in range(1, len(inputs)):
            corr_feat = self.corrs[level](inputs[t], inputs[t-1])
            corr_feats.append(corr_feat)
        corr_feats = torch.cat(corr_feats)
        return corr_feats

