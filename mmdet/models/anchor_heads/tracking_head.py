import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (AnchorGenerator, multi_apply,
                        anchor_target_tracking)
from mmdet.models.backbones.ssd_mobilenet_v2 import ExtraConv
from mmdet.ops import Correlation
from mmdet.core.bbox import bbox2delta
from ..losses import smooth_l1_loss
from ..registry import HEADS
from .anchor_head import AnchorHead


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
                 strides=(2, 1, 1)):
        super(DnTLayer, self).__init__()
        self.point_corrs = nn.ModuleList([
            Correlation(pad_size=disp, kernel_size=1, max_displacement=disp,
                                 stride1=s, stride2=s)
            for disp, s in zip(displacements, strides)
        ])
        corr_chans = sum([
            (2 * (d // s) + 1)**2
            for d, s in zip(displacements, strides)])
        self.base_conv = ExtraConv(corr_chans, 256, stride=1, insert_1x1_conv=True)
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
        """

        Args:
            feats: [[T, B, C, H, W} x 3]

        Returns:
            multi-level [(T-1)*B, C, H, W] x 4
        """
        c3, c4, c5 = feats
        interp_func = partial(F.interpolate, size=(19, 19), mode='nearest')
        corr_feats = []
        for t in range(0, c3.size(0) - 1):
            c3_corr = self.point_corrs[0](c3[t], c3[t+1])
            c4_corr = self.point_corrs[1](c4[t], c4[t+1])
            c5_cur = interp_func(c5[t])
            c5_nxt = interp_func(c5[t+1])
            c5_corr = self.point_corrs[2](c5_cur, c5_nxt)
            corr_feat = [c3_corr, c4_corr, c5_corr]
            corr_feat = torch.cat(corr_feat, dim=1)  # cat c3, c4, c5
            corr_feats.append(corr_feat)
        corr_feats = torch.cat(corr_feats, dim=0)

        corr_feats = self.base_conv(corr_feats)
        multilevel_feats = [corr_feats]
        x = corr_feats
        for conv in self.extra:
            x = conv(x)
            multilevel_feats.append(x)
        return multilevel_feats


# TODO: add loss evaluator for SSD
@HEADS.register_module
class TrackingHead(AnchorHead):

    def __init__(self,
                 loss_scale=1.0,
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.loss_scale = loss_scale
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        # num_anchors = [4, 6, 6, 6, 4, 4], (if 1 then 4, if 2 then 6)
        track_convs = []
        for i in range(len(in_channels)):
            track_convs.append(
                nn.Conv2d(
                    256,  # TODO(zhifan) fix this 256 hack
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
        self.track_convs = nn.ModuleList(track_convs)
        self.dnt_layer = DnTLayer()

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
        self.dnt_layer.init_weights()

    def forward(self, feats):
        feats = self.dnt_layer(feats)
        # feats is now [(T-1)*B, C, H ,W) x layers]

        track_preds = []
        for feat, track_conv in zip(feats, self.track_convs):
            track_preds.append(track_conv(feat))
        return track_preds

    def loss_single(self, track_pred, track_targets, track_weights,
                    num_total_samples, cfg):
        loss_track = self.loss_scale * smooth_l1_loss(
            track_pred,
            track_targets,
            track_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return (loss_track, )

    @staticmethod
    def compute_track_delta(gt_bboxes,
                            gt_trackids,
                            seq_len):
        """

        Args:
            gt_bboxes: list, T*B x [N, 4]
            gt_trackids: list, T*B x [N, ]
            seq_len: int

        Returns:
            gt_bboxes_keep_t1: list, (T-1)*B x [M, 4], keep this to guide assigner.
            gt_track_delta: list, (T-1)*B x [M, 4] for t1

        """
        batch = len(gt_bboxes) // seq_len
        gt_bboxes_keep_curr = []
        gt_track_delta = []
        for t in range(0, seq_len - 1):
            gt_bboxes_curr = gt_bboxes[t*batch:(t+1)*batch]
            gt_bboxes_next = gt_bboxes[(t+1)*batch:(t+2)*batch]
            gt_trackids_curr = gt_trackids[t*batch:(t+1)*batch]
            gt_trackids_next = gt_trackids[(t+1)*batch:(t+2)*batch]
            for bb_t1, bb_t2, tid_t1, tid_t2 in zip(
                    gt_bboxes_curr, gt_bboxes_next,
                    gt_trackids_curr, gt_trackids_next):
                tid1_list = tid_t1.tolist()
                tid2_list = tid_t2.tolist()
                exist_both = set(tid1_list).intersection(set(tid2_list))
                ind_t1 = [tid1_list.index(v) for v in exist_both]
                ind_t2 = [tid2_list.index(v) for v in exist_both]
                bb_t1 = bb_t1[ind_t1]
                bb_t2 = bb_t2[ind_t2]
                gt_bb_delta = bbox2delta(bb_t1, bb_t2)  # mean 0, std 1
                gt_track_delta.append(gt_bb_delta)
                gt_bboxes_keep_curr.append(bb_t1)

        return gt_bboxes_keep_curr, gt_track_delta

    def loss(self,
             seq_len,
             cls_scores,
             track_preds,
             gt_bboxes,
             gt_trackids,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        """
        During loss forwarding, some frame may have zero gt_bbox, and some frame
        may have
            cls_scores: [T*B, C, H, W], We need this for feat_map_size
            track_preds: [(T-1)*B, C, H, W]
            gt_bboxes: [T*B, 4]
            gt_trackids: [T*B, ]
            img_metas: [T*B]
            cfg:
            gt_bboxes_ignore:

        Returns:

        """
        # Extract first T-1 frames
        batch = cls_scores[0].size(0) // seq_len
        cls_scores = [c[:batch*(seq_len-1)] for c in cls_scores]
        img_metas = img_metas[:batch * (seq_len - 1)]
        gt_bboxes_keep_t0, gt_deltas_list = self.compute_track_delta(
            gt_bboxes, gt_trackids, seq_len)

        # Some frames may have no track, disable calculation on them.
        valid_gt_ind = [i for i, gt in enumerate(gt_bboxes_keep_t0) if len(gt) != 0]
        if len(valid_gt_ind) == 0:
            return dict(
                loss_track=torch.tensor(
                    0.0,
                    dtype=track_preds[0].dtype,
                    device=track_preds[0].device))
        cls_scores = [c[valid_gt_ind, ...].contiguous() for c in cls_scores]
        track_preds = [tr[valid_gt_ind, ...].contiguous() for tr in track_preds]
        gt_bboxes_keep_t0 = [gt_bboxes_keep_t0[i] for i in valid_gt_ind]
        gt_deltas_list = [gt_deltas_list[i] for i in valid_gt_ind]
        img_metas = [img_metas[i] for i in valid_gt_ind]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        # Compute Track targets
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        track_targets = anchor_target_tracking(
            anchor_list,
            valid_flag_list,
            gt_bboxes_list=gt_bboxes_keep_t0,
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
            self.loss_single,
            all_track_preds,
            all_track_targets,
            all_track_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)
        losses_track = losses_track[0]

        return dict(loss_track=losses_track)
