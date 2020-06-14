import torch
import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .track_base import TrackBaseDetector

"""
The difference with PairSingleStage is:
    At train time, TrackDetector takes two input and calculate loss for BOTH.
    PairDetector/TwinDetector calculate loss for ONE only
    
    At test time, both takes one input.

This class is designed for RetinaTrack (NOT include D & T, as D&T doesn't produce 
third result from detector head)
"""


@DETECTORS.register_module
class TrackSingleStageDetector(TrackBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TrackBaseDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(TrackSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      ref_img,
                      img_metas,
                      ref_img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_trackids,
                      ref_bboxes,
                      ref_labels,
                      ref_trackids,
                      gt_bboxes_ignore=None):
        losses = dict()

        img = torch.cat([img, ref_img], dim=0)
        img_metas = img_metas + ref_img_metas
        gt_bboxes = gt_bboxes + ref_bboxes
        gt_labels = gt_labels + ref_labels
        gt_trackids = gt_trackids + ref_trackids

        x = self.extract_feat(img)  # [ [2*B, C, H, W] x #fpn ]
        cls, reg, embed = self.bbox_head(x)  # (3 [#fpn, [B, _, H_i, W_i] ] )

        loss_inputs = (cls, reg, gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses_det = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(losses_det)

        embed_loss_input = (embed, gt_bboxes, gt_trackids, img_metas, self.train_cfg)
        embed_losses = self.bbox_head.embed_loss(
            *embed_loss_input, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(embed_losses)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, _ in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
