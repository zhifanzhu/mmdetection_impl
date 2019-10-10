import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .seq_base import SeqBaseDetector


@DETECTORS.register_module
class SeqRNNSSDMobileNet(SeqBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 temporal_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SeqRNNSSDMobileNet, self).__init__()
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
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SeqRNNSSDMobileNet, self).init_weights(pretrained)
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

    def extract_feat(self, img):
        """ Extract base features. """
        x = self.backbone.extract_feats(img, endpoint='layer15')
        if self.with_neck and self.neck_first:
            x = self.neck(x)
        return x

    def extract_final_feat(self, feats):
        """ Extract feature from **layer15** and up.

        Args:
            feats: tuple of Tensor

        Returns:
            tuple, similar to output of extract_feats.
        """
        # layer15_cont: The leftover of layer15, (BN, etc)
        # features_cont: The left over of MobileNet.features
        layer15_cont = self.backbone.features[14].conv[1:]
        features_cont = self.backbone.features[15:]

        x = layer15_cont(feats[-1])
        x = features_cont(x)
        feats.append(x)

        for i, layer in enumerate(self.backbone.extra):
            x = layer(x)
            feats.append(x)
        return tuple(feats)

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
                      gt_bboxes_ignore=None):
        batch = img.size(0) // seq_len
        x = self.extract_feat(img)  # [[T*B, c1, h1, w1]*4]
        if self.with_temporal_module:
            x_seq = [v.view([seq_len, batch, *v.shape[1:]])
                     for v in x]
            x, _ = self.temporal_module(x_seq, in_dict=None, is_train=True)
            # x = [[T*B, c, h, *w]*5]

        x = self.extract_final_feat(x)
        if self.with_neck and not self.neck_first:
            x = self.neck(x)

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
        img_metas = [img_metas[i] for i in valid_gt_ind]

        loss_inputs = new_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def temporal_test(self, img, img_meta, seq_len, rescale=False):
        x = self.extract_feat(img)  # [[1*1, c1, h1, w1]*4]
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
