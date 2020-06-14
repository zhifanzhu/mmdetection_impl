import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from mmdet.core import (delta2bbox, force_fp32, anchor_target,
                        multiclass_nms_with_feat)
from ..builder import build_loss
from .anchor_head import AnchorHead

"""
This is the complete implementation of RetinaTrack with embed feature and support
training with triplet loss.
See Also retina_track_head.py.

Naming convention (hopefully):
    `embed` is associated to features and loss
    `triplet` is for specific loss function
    `contrastive` is for Class name
"""


@HEADS.register_module
class RetinaContrastiveHead(AnchorHead):
    """ See [RetinaTrack]
    """

    def __init__(self,
                 m1,
                 m2,
                 m3,
                 num_classes,
                 in_channels,
                 num_triplets=64,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 freeze_all=False,
                 **kwargs):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.num_triplets = num_triplets
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.freeze_all = freeze_all
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        loss_embed = kwargs.pop('loss_embed')
        super(RetinaContrastiveHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

        self.loss_embed = build_loss(loss_embed)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)

        # Task shared
        self.m1_convs = nn.ModuleList()
        for i in range(self.m1):
            chn = self.in_channels if i == 0 else self.feat_channels
            chn = chn * self.num_anchors
            self.m1_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels * self.num_anchors,
                    3,
                    stride=1,
                    padding=1,
                    groups=self.num_anchors,  # num_anchors in parallel
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.m2):
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.embed_convs = nn.ModuleList()
        for i in range(self.m3):
            self.embed_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)

        if self.freeze_all:
            def _freeze_conv(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    m.requires_grad = False
            self.apply(_freeze_conv)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        bsize, _, height, width = x.shape
        feat = x.repeat(1, self.num_anchors, 1, 1)
        for i in range(self.m1):
            feat = self.m1_convs[i](feat)
        task_shared_feat = feat.view(
            bsize * self.num_anchors, self.feat_channels, height, width)

        cls_feat = task_shared_feat
        reg_feat = task_shared_feat
        for i in range(self.m2):
            cls_feat = self.cls_convs[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_feat)

        for i in range(self.m3):
            task_shared_feat = self.embed_convs[i](task_shared_feat)
        embed_feat_shape = (
            bsize, self.num_anchors, self.feat_channels, height, width)

        cls_score = self.retina_cls(cls_feat).view(
            bsize, self.cls_out_channels * self.num_anchors, height, width)
        bbox_pred = self.retina_reg(reg_feat).view(
            bsize, 4 * self.num_anchors, height, width)
        return cls_score, bbox_pred, task_shared_feat.view(*embed_feat_shape)

    def embed_loss_single(self, embed_feats, track_targets):
        """ embed_Feat: [2*B, #Anchor, feat_chan, H, W] -> [2*B, #Anchor, H, W, feat_chan]
                       -> [2*B, #Anchor*H*W, feat_chan]
            where the 2 in batch dimension include 'img' and 'ref_img'

        Note some FPN level has ZERO of tracks.
        """
        batch_size = len(track_targets) // 2
        embed_feats = embed_feats.permute(0, 1, 3, 4, 2).reshape(
            2 * batch_size, -1, self.feat_channels)

        ancs = []
        poss = []
        negs = []
        for i in range(batch_size):
            num_tracks = max(track_targets[i].max(), track_targets[i + batch_size].max())
            if num_tracks == 0:
                continue

            # [2*#Anchor*H*W, feat_chan]
            embed_feat = torch.cat(
                [embed_feats[i], embed_feats[i + batch_size]], 0)
            track_target = torch.cat(
                [track_targets[i], track_targets[i + batch_size]], 0)  # [2*H*W*#Anchor]

            # index 0 is for background, which we clearly don't want to index into
            for j in range(1, num_tracks + 1):
                all_pos_inds = (track_target == j).nonzero()
                all_neg_inds = (track_target != j).nonzero()
                num_all_pos = all_pos_inds.size(0)
                num_triplets_per_track = min(
                    num_all_pos, self.num_triplets // num_tracks)
                if num_all_pos == 0:
                    continue
                anc_ind_inds = torch.randperm(num_all_pos)[:num_triplets_per_track]
                neg_ind_inds = torch.randperm(all_neg_inds.size(0))[:num_triplets_per_track]
                pos_ind_inds = neg_ind_inds.new_zeros(
                    neg_ind_inds.shape, dtype=torch.long)

                # We need `pos` distinct from `anc`
                if anc_ind_inds.size(0) < 2:
                    continue
                for k in range(num_triplets_per_track):
                    anc_ind_ind = anc_ind_inds[k]
                    pos_ind_ind = torch.randint(0, num_all_pos, (1, ))  # [int]
                    # How is the speed? Will it stuck?
                    while pos_ind_ind == anc_ind_ind:
                        pos_ind_ind = torch.randint(0, num_all_pos, (1, ))
                    pos_ind_inds[k] = pos_ind_ind

                anc_inds = all_pos_inds[anc_ind_inds]
                pos_inds = all_pos_inds[pos_ind_inds]
                neg_inds = all_neg_inds[neg_ind_inds]
                anc = embed_feat[anc_inds]
                pos = embed_feat[pos_inds]
                neg = embed_feat[neg_inds]
                ancs.append(anc)
                poss.append(pos)
                negs.append(neg)

        if len(ancs) == 0:
            return embed_feats.new_zeros([])
        else:
            ancs = torch.cat(ancs)
            poss = torch.cat(poss)
            negs = torch.cat(negs)
            loss_embed = self.loss_embed(ancs, poss, negs)
            return loss_embed

    @force_fp32(apply_to=('embed_feats_list', ))
    def embed_loss(self,
                   embed_feats_list,
                   gt_bboxes,
                   gt_trackids,
                   img_metas,
                   cfg,
                   gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in embed_feats_list]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = embed_feats_list[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cache_assigner = cfg.assigner
        cfg.assigner = cfg.track_assigner
        embed_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_trackids,
            label_channels=label_channels,
            sampling=self.sampling)
        cfg.assigner = cache_assigner
        if embed_targets is None:
            return None
        (track_targets, track_targets_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = embed_targets
        losses_triplet = list(map(
            self.embed_loss_single, *(embed_feats_list, track_targets)))
        return dict(loss_triplet=losses_triplet)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'embed_feats'))
    def get_bboxes(self, cls_scores, bbox_preds, embed_feats, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            embed_feats_list = [
                embed_feats[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               embed_feats_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          embed_feats_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_feats = []
        for cls_score, bbox_pred, feat, anchors in zip(cls_score_list,
                                                       bbox_pred_list,
                                                       embed_feats_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # Feat: [#A, C, H, W] -> [#A*H*W, C]
            feat = feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                feat = feat[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_feats.append(feat)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_feats = torch.cat(mlvl_feats)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels, det_feats = multiclass_nms_with_feat(
            mlvl_bboxes, mlvl_scores, mlvl_feats, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels, det_feats

