import torch
import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .pair_base import PairBaseDetector

from mmdet.models import build_detector
from mmcv.runner.checkpoint import load_checkpoint


@DETECTORS.register_module
class TwinTriSingleStageDetector(PairBaseDetector):

    def __init__(self,
                 backbone,
                 twin,
                 twin_load_from,
                 middle='B',  # 'S'
                 neck=None,
                 bbox_head=None,
                 pair_module=None,
                 pair_module2=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwinTriSingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if pair_module is not None:
            self.pair_module = builder.build_pair_module(
                pair_module)
        if pair_module2 is not None:
            self.pair_module2 = builder.build_pair_module(
                pair_module2)

        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # Build twin model
        print(" Loading Twin's weights...")
        self.twin = build_detector(twin, train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        # self.twin = MMDataParallel(twin, device_ids=[0]).cuda()  # TODO check device id?
        load_checkpoint(self.twin, twin_load_from, map_location='cpu', strict=False, logger=None)
        self.twin.eval()
        self.middle = middle

        # memory cache for testing
        self.test_interval = test_cfg.get('test_interval', 10)
        self.key_feat_pre = None
        self.key_feat_post = None

    def init_weights(self, pretrained=None):
        super(TwinTriSingleStageDetector, self).init_weights(pretrained)
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
        x = self.neck(x)
        return x

    def forward_dummy(self, img):
        # TODO zhifan, for mmdetection/tools/get_flops.py
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self, img, img_metas, **kwargs):
        if self.middle == 'B':
            return self.forward_train_b(img, img_metas, **kwargs)
        elif self.middle == 'S':
            return self.forward_train_s(img, img_metas, **kwargs)
        else:
            raise ValueError

    def forward_train_b(self,
                        img,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        first_img,
                        second_img,
                        gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        with torch.no_grad():
            x_first = self.twin.extract_feat(first_img)
            x_second = self.twin.extract_feat(second_img)
            x_second = self.twin.pair_module(x_second, x_first)

        x = self.pair_module(x, x_second)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_train_s(self,
                        img,
                        img_metas,
                        first_img,
                        second_img,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        with torch.no_grad():
            x_first = self.twin.extract_feat(first_img)

        x_second = self.twin.extract_feat(second_img)
        x_second = self.pair_module(x_first, x_second)
        x = self.pair_module2(x, x_second)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, **kwargs):
        if self.middle == 'B':
            return self.simple_test_b(**kwargs)
        elif self.middle == 'S':
            return self.simple_test_s(**kwargs)
        else:
            raise ValueError

    def simple_test_b(self, img, img_meta, rescale=False):
        frame_ind = img_meta[0]['frame_ind']
        if frame_ind == 0:
            twin = self.twin
            x = twin.extract_feat(img)
            x_cache = x
            outs = twin.bbox_head(x)
            bbox_inputs = outs + (img_meta, twin.test_cfg, rescale)
            bbox_list = twin.bbox_head.get_bboxes(*bbox_inputs)

            self.key_feat_pre = x_cache
            self.key_feat_post = x_cache

        elif (frame_ind % self.test_interval) == 0:
            twin = self.twin
            x = twin.extract_feat(img)
            x_cache = x
            x = self.twin.pair_module(x, self.key_feat_pre)
            outs = twin.bbox_head(x)
            bbox_inputs = outs + (img_meta, twin.test_cfg, rescale)
            bbox_list = twin.bbox_head.get_bboxes(*bbox_inputs)

            self.key_feat_pre = x_cache
            self.key_feat_post = x

        else:
            x = self.extract_feat(img)
            x = self.pair_module(x, self.key_feat_post, is_train=False)
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def simple_test_s(self, img, img_meta, rescale=False):
        frame_ind = img_meta[0]['frame_ind']
        if (frame_ind % self.test_interval) == 0:
            twin = self.twin
            x = twin.extract_feat(img)
            outs = twin.bbox_head(x)
            bbox_inputs = outs + (img_meta, twin.test_cfg, rescale)
            bbox_list = twin.bbox_head.get_bboxes(*bbox_inputs)

            self.key_feat_pre = x

        elif ((frame_ind - 1) % self.test_interval) == 0:
            x = self.extract_feat(img)
            x = self.pair_module(x, self.key_feat_pre, is_train=False)
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

            self.key_feat_post = x

        else:
            x = self.extract_feat(img)
            x = self.pair_module(x, self.key_feat_post, is_train=False)
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
