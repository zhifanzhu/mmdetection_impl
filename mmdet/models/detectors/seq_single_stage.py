import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .seq_base import SeqBaseDetector


@DETECTORS.register_module
class SeqSingleStageDetector(SeqBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 temporal_module=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SeqSingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if temporal_module is not None:
            self.temporal_module = builder.build_temporal_module(
                temporal_module)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SeqSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        # assert len(img.shape) == 4
        batch, time, chan, height, width = img.shape
        img = img.reshape([batch*time, chan, height, width])
        x = self.backbone(img)  # [[b*t, c1, h1, w1]*4]
        if self.with_neck:
            x = self.neck(x)
        if self.with_temporal_module:
            x = [v.reshape([batch, time, *v.shape[1:]])
                 for v in x]
            x_seq = [v.permute([1, 0, 2, 3, 4]) for v in x]  # [[t, b, c1, h1, w1]*4]
            x = self.temporal_module(x_seq)  # [[b*t, c, h, *w]*4]
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # TODO(zhifan) determine losses shape, reshape back here?
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
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
