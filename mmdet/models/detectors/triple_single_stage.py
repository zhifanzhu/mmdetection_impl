import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .triple_base import TripleBaseDetector


@DETECTORS.register_module
class TripleSingleStageDetector(TripleBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 triple_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TripleSingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.neck_first = True
        if triple_module is not None:
            self.triple_module = builder.build_triple_module(
                triple_module)
            if hasattr(self.triple_module, 'neck_first'):
                self.neck_first = self.triple_module.neck_first
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # memory cache for testing
        self.prev_memory = None

    def init_weights(self, pretrained=None):
        super(TripleSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_triple_module:
            self.triple_module.init_weights()
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
                      near_img,
                      far_img,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        x_near = self.extract_feat(near_img)
        x_far = self.extract_feat(far_img)
        x = self.triple_module(x, x_near, x_far, is_train=True)
        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    # def simple_test(self, img, img_meta, rescale=False):
    #     x = self.extract_feat(img)
    #     x_cache = x
    #     is_first = img_meta[0]['is_first']
    #     if is_first:
    #         x = x
    #     else:
    #         x = self.triple_module(x, self.prev_memory, is_train=False)
    #
    #     if self.with_neck and not self.neck_first:
    #         x = self.neck(x)
    #
    #     # self.prev_memory = x
    #     self.prev_memory = x_cache
    #
    #     outs = self.bbox_head(x)
    #     bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #     bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results[0]

    def simple_test(self, img, img_meta, near_img, far_img, rescale=False):
        x = self.extract_feat(img)
        x_near = self.extract_feat(near_img)
        x_far = self.extract_feat(far_img)
        x = self.triple_module(x, x_near, x_far, is_train=False)

        if self.with_neck and not self.neck_first:
            x = self.neck(x)

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
