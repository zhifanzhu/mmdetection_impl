import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .pair_base import PairBaseDetector


@DETECTORS.register_module
class PairSingleStageDetector(PairBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 pair_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PairSingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.neck_first = True
        if pair_module is not None:
            self.pair_module = builder.build_pair_module(
                pair_module)
            if hasattr(self.pair_module, 'neck_first'):
                self.neck_first = self.pair_module.neck_first
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # memory cache for testing
        self.prev_memory = None

    def init_weights(self, pretrained=None):
        super(PairSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_temporal_module:
            self.pair_module.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck and self.neck_first:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        # TODO zhifan, for mmdetection/tools/get_flops.py
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      ref_img,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        x_ref = self.extract_feat(ref_img)
        x = self.pair_module(x, x_ref, is_train=True)
        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, img_prev, rescale=False):
        """ img_prev is passed in, but we may choose to not use it.
        TODO(zhifan) delete img_prev for optimization?
        """
        x = self.extract_feat(img)
        is_first = img_meta['is_first']
        if not is_first:
            x = self.pair_module(x, self.prev_memory, is_train=False)

        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        self.prev_memory = x

        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError