import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result
from mmdet.models.backbones.ssd_mobilenet_v2 import ExtraConv
from mmdet.ops import PointwiseCorrelation
from .. import builder
from ..registry import DETECTORS
from .seq_base import SeqBaseDetector


class DnTLayer(nn.Module):
    """
    During forward, DnT Layer receives [(32, 38, 38), (96, 19, 19), (1280, 10, 10)],
    it performs stride 2 point corr on 38x38, normal point corr on 19x19 ,
    and nearest upsampling on 10x10 then point corr.

    Displacements are (4, 4, 2), covers (9, 9, 5), and produces (81, 81, 25)
    for each layer, i.e. 2*d+1, then concat to produce (187, 19, 19)

    These output will then be fed into subsequent layers to produce
    multi-level feat map,
    (THIS WILL be done in ssd-dnt-head and finally fed to ssd-head like reg head.)

    """

    def __init__(self,
                 displacements=(4, 4, 4),
                 strides=(1, 1, 1)):
        super(DnTLayer, self).__init__()
        self.point_corrs = nn.ModuleList([
            PointwiseCorrelation(disp, s)
            for disp, s in zip(displacements, strides)
        ])
        self.base_conv = ExtraConv(243, 256, stride=1, insert_1x1_conv=True)
        # One more layer for 19x19 -> 10x10
        self.extra = nn.ModuleList([
            ExtraConv(256, 256, stride=2, insert_1x1_conv=True),
            ExtraConv(256, 256, stride=2, insert_1x1_conv=True),
            ExtraConv(256, 256, stride=2, insert_1x1_conv=True),
            ExtraConv(256, 256, stride=2, insert_1x1_conv=True),
            ExtraConv(256, 256, stride=2, insert_1x1_conv=True),
        ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, feats):
        c3, c4, c5 = feats
        c3_t0 = F.interpolate(c3[0], size=(19, 19), mode='nearest')
        c3_t1 = F.interpolate(c3[1], size=(19, 19), mode='nearest')
        c3_corr = self.point_corrs[0](c3_t0, c3_t1)  # (cur, next)
        c4_corr = self.point_corrs[1](c4[0], c4[1])
        # c5 = F.interpolate(c5.clone(), scale_factor=2, mode='nearest')
        c5_t0 = F.interpolate(c5[0], size=(19, 19), mode='nearest')
        c5_t1 = F.interpolate(c5[1], size=(19, 19), mode='nearest')
        c5_corr = self.point_corrs[2](c5_t0, c5_t1)
        corr_feats = torch.cat([
            # (b, H, W, (2d+1), (2d+1)) -> (b, (2d+1)^2, H, W)
            cf.view(cf.size(0), cf.size(1), cf.size(2), -1).permute(0, 3, 1, 2)
            for cf in [c3_corr, c4_corr, c5_corr]
        ], dim=1)

        corr_feats = self.base_conv(corr_feats)
        multilevel_feats = [corr_feats]
        x = corr_feats
        for conv in self.extra:
            x = conv(x)
            multilevel_feats.append(x)
        return multilevel_feats


@DETECTORS.register_module
class SeqSSDDnT(SeqBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 temporal_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SeqSSDDnT, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.neck_first = True
        if temporal_module is not None:
            self.temporal_module = builder.build_temporal_module(
                temporal_module)
            if hasattr(self.temporal_module, 'neck_first'):
                self.neck_first = self.temporal_module.neck_first
        self.bbox_head = builder.build_head(bbox_head)
        self.dnt_layer = DnTLayer()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SeqSSDDnT, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_temporal_module:
            self.temporal_module.init_weights()
        self.bbox_head.init_weights()
        self.dnt_layer.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck and self.neck_first:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        # TODO zhifan
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      seq_len,
                      gt_bboxes,
                      gt_labels,
                      gt_trackids,
                      gt_bboxes_ignore=None):
        assert seq_len == 2
        batch = img.size(0) // seq_len
        raw_x = self.extract_feat(img)  # [[2*B, c1, h1, w1]*{7, 14, 15, 19, &extra}]
        x = raw_x[2:]
        if self.with_temporal_module:
            x_seq = [v.view([seq_len, batch, *v.shape[1:]])
                     for v in x]
            x, _ = self.temporal_module(x_seq, in_dict=None, is_train=True)

        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        # Predice track
        track_x_seq = [v.view([seq_len, batch, *v.shape[1:]])
                       for i, v in enumerate(raw_x) if i in (0, 1, 3)]
        track_feats = self.dnt_layer(track_x_seq)
        track_preds = self.bbox_head.forward_track(track_feats)

        outs = self.bbox_head(x)

        # Some frames may have no annotation, disable calculation on them.
        valid_gt_ind = [i for i, gt in enumerate(gt_bboxes) if len(gt) != 0]
        new_outs = []
        for out in outs:
            new_out = [o[valid_gt_ind, ...].contiguous()
                       for o in out]
            new_outs.append(new_out)
        new_outs = tuple(new_outs)
        gt_bboxes = [gt_bboxes[i] for i in valid_gt_ind]
        gt_labels = [gt_labels[i] for i in valid_gt_ind]
        gt_trackids = [gt_trackids[i] for i in valid_gt_ind]
        img_metas = [img_metas[i] for i in valid_gt_ind]

        loss_inputs = new_outs + (track_preds, gt_bboxes, gt_labels,
                                  gt_trackids, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def temporal_test(self, img, img_meta, seq_len, rescale=False):
        x = self.extract_feat(img)  # [[1*1, c1, h1, w1]*4]
        x = x[2:]
        out_dict = None
        if self.with_temporal_module:
            x_seq = [v.view([seq_len, 1, *v.shape[1:]])
                     for v in x]
            x, out_dict = self.temporal_module(x_seq, in_dict=None, is_train=True)

        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results, out_dict

    def simple_test(self, img, img_meta, in_dict=None, rescale=False):
        x = self.extract_feat(img)  # [[1*1, c1, h1, w1]*5]
        out_dict = None
        if self.with_temporal_module:
            # During test, no reshape & permute
            x, out_dict = self.temporal_module(x, in_dict=in_dict,
                                               is_train=False)

        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0], out_dict

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
