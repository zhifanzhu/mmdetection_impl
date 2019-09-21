import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (AnchorGenerator, anchor_target, multi_apply,
                        anchor_target_tracking)
from mmdet.core.bbox import bbox2delta
from ..losses import smooth_l1_loss
from ..registry import HEADS
from .anchor_head import AnchorHead


# TODO: add loss evaluator for SSD
@HEADS.register_module
class SSDDnTHead(AnchorHead):

    def __init__(self,
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        # num_anchors = [4, 6, 6, 6, 4, 4], (if 1 then 4, if 2 then 6)
        reg_convs = []
        cls_convs = []
        track_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
            track_convs.append(
                nn.Conv2d(
                    256,
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)
        self.track_convs = nn.ModuleList(track_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]  # Typical value: [1., 1.414]
            ratios = [1.]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.fp16_enabled = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def forward_track(self, feats):
        track_preds = []
        for feat, track_conv in zip(feats, self.track_convs):
            track_preds.append(track_conv(feat))
        return track_preds

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss_track(self, track_pred, track_targets, track_weights,
                   num_total_samples, cfg):
        loss_track = smooth_l1_loss(
            track_pred,
            track_targets,
            track_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return (loss_track, )

    def compute_track_delta(self,
                    gt_bboxes,
                    gt_trackids):
        """

        Args:
            gt_bboxes: list, T*B x [N, 4]
            gt_trackids: list, T*B x [N, ]

        Returns:
            gt_bboxes_keep_t1: list, 1*B x [M, 4], keep this to guide assigner.
            gt_track_delta: list, 1*B x [M, 4] for t1

        """
        # Assume time dim == 2
        batch = len(gt_bboxes) // 2
        gt_bboxes_t1 = gt_bboxes[:batch]
        gt_bboxes_t2 = gt_bboxes[batch:]
        gt_trackids_t1 = gt_trackids[:batch]
        gt_trackids_t2 = gt_trackids[batch:]
        gt_bboxes_keep_t1 = []
        gt_track_delta = []
        for bb_t1, bb_t2, tid_t1, tid_t2 in zip(
            gt_bboxes_t1, gt_bboxes_t2, gt_trackids_t1, gt_trackids_t2):
            tid1_list = tid_t1.tolist()
            tid2_list = tid_t2.tolist()
            exist_both = set(tid1_list).intersection(set(tid2_list))
            ind_t1 = [tid1_list.index(v) for v in exist_both]
            ind_t2 = [tid2_list.index(v) for v in exist_both]
            bb_t1 = bb_t1[ind_t1]
            bb_t2 = bb_t2[ind_t2]
            gt_bb_delta = bbox2delta(bb_t1, bb_t2)  # mean 0, std 1
            gt_track_delta.append(gt_bb_delta)
            gt_bboxes_keep_t1.append(bb_t1)
        return gt_bboxes_keep_t1, gt_track_delta

    def loss(self,
             cls_scores,
             bbox_preds,
             track_preds,
             gt_bboxes,
             gt_labels,
             gt_trackids,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        # Compute normal cls and reg
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            sampling=False,
            unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)

        # Compute Track targets
        batch = cls_scores[0].size(0) // 2
        img_metas = img_metas[:batch]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        # anchor_list = anchor_list[:batch]
        # valid_flag_list = valid_flag_list[:batch]
        gt_bboxes_keep_t1, gt_deltas_list = self.compute_track_delta(
            gt_bboxes, gt_trackids)
        track_targets = anchor_target_tracking(
            anchor_list,
            valid_flag_list,
            gt_bboxes_list=gt_bboxes_keep_t1,
            gt_deltas_list=gt_deltas_list,
            img_metas=img_metas,
            cfg=cfg,
            sampling=False,
            unmap_outputs=False)
        (track_targets_list, track_weights_list,
         num_total_pos, num_total_neg) = track_targets
        num_images = len(img_metas)
        all_track_preds = torch.cat([
            tr.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for tr in track_preds
        ], -2)
        all_track_targets = torch.cat(track_targets_list,
                                      -2).view(num_images, -1, 4)
        all_track_weights = torch.cat(track_weights_list,
                                      -2).view(num_images, -1, 4)
        losses_track = multi_apply(
            self.loss_track,
            all_track_preds,
            all_track_targets,
            all_track_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)
        losses_track = losses_track[0]

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_track=losses_track)
