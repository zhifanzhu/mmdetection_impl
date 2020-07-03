import torch
import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .pair_base import PairBaseDetector

from mmdet.models import build_detector
from mmcv.runner.checkpoint import load_checkpoint


@DETECTORS.register_module
class TwinV2SingleStageDetector(PairBaseDetector):

    def __init__(self,
                 backbone,
                 twin,
                 twin_load_from,
                 neck=None,
                 bbox_head=None,
                 pair_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwinV2SingleStageDetector, self).__init__()
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

        # Build twin model
        print(" Loading Twin's weights...")
        self.twin = build_detector(twin, train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        # self.twin = MMDataParallel(twin, device_ids=[0]).cuda()  # TODO check device id?
        load_checkpoint(self.twin, twin_load_from, map_location='cpu', strict=False, logger=None)
        print(" Finished loading twin's weight. ")
        self.twin.eval()

        # memory cache for testing
        self.key_feat = None

    def init_weights(self, pretrained=None):
        super(TwinV2SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_pair_module:
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
        with torch.no_grad():
            x_ref = self.twin.extract_feat(ref_img)
        x = self.pair_module(x, x_ref, is_train=True)
        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        frame_ind = img_meta[0]['frame_ind']
        is_key = img_meta[0]['is_key']
        if frame_ind == 0:
            twin = self.twin
            x = twin.extract_feat(img)
            outs = twin.bbox_head(x)
            bbox_inputs = outs + (img_meta, twin.test_cfg, rescale)
            bbox_list = twin.bbox_head.get_bboxes(*bbox_inputs)

            self.key_feat = x
        else:
            if is_key:
                twin = self.twin
                x = twin.extract_feat(img)
                x = twin.pair_module(x, self.key_feat)
                outs = twin.bbox_head(x)
                bbox_inputs = outs + (img_meta, twin.test_cfg, rescale)
                bbox_list = twin.bbox_head.get_bboxes(*bbox_inputs)

                self.key_feat = x
            else:
                x = self.extract_feat(img)
                x = self.pair_module(x, self.key_feat, is_train=False)
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
