import torch
import torch.nn as nn
import copy
import numpy as np

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
from mmdet.core.bbox import bbox_overlaps


class ScoredPath(object):
    def __init__(self, score, ind, vec_prev):
        self.score_list = [score]
        self.ind_list = [ind]
        self.vec_prev = vec_prev
        self.cnt = 0

    def update(self, score, ind, vec_cur):
        """ Update track as well as current vec"""
        self.score_list.append(score)
        self.ind_list.append(ind)
        vec_cur = (vec_cur + self.vec_prev * self.cnt) / (1 + self.cnt)
        self.cnt += 1
        self.vec_prev = vec_cur
        return vec_cur

    def __len__(self):
        return len(self.ind_list)

    def __repr__(self):
        return 'ScoredPath: ' + repr(self.ind_list)

    def assign_new_last(self):
        return torch.mean(torch.Tensor(self.score_list))


def create_matrix(I, J):
    """

    Args:
        I: [N, 4 + 1 + vec_dim], Tensor
        J: [M, 4 + 1 + vec_dim], Tensor
    Returns:
        [N, M]
    """
    iou = bbox_overlaps(I[:, :4], J[:, :4])

    def _similarity(a, b):
        a_norm = a / a.norm(dim=1, keepdim=True)
        b_norm = b / b.norm(dim=1, keepdim=True)
        return torch.matmul(a_norm, b_norm.transpose(1, 0))
    prod = _similarity(I[:, 5:], J[:, 5:])
    mat = 1. / (prod * iou)
    return mat


def match_and_pairs(prev, cur):
    num_cols = len(cur)
    if len(prev) == 0:
        return []
    if len(cur) == 0:
        return []
    dist_mat = create_matrix(prev, cur)
    pairs = []
    while not torch.isinf(dist_mat.min()):
        min_ind = dist_mat.argmin()
        i, j = min_ind // num_cols, min_ind % num_cols
        pairs.append((i, j))
        dist_mat[i, :] = float('inf')
        dist_mat[:, j] = float('inf')
    return pairs


def update_tublets(prev_dets, cur_dets, prev_tublets):
    if prev_tublets is None and prev_dets is None:
        cur_tublets = []
        for c in range(30):
            cur_tublet = [ScoredPath(
                d[4], i, d[5:]) for i, d in enumerate(cur_dets[c])]
            cur_tublets.append(cur_tublet)
        return cur_dets, cur_tublets

    cur_tublets = []
    for c in range(30):
        prev_det = prev_dets[c]
        cur_det = cur_dets[c]
        pairs = match_and_pairs(prev_det, cur_det)

        cur_tublet = [ScoredPath(
            d[4], i, d[5:]) for i, d in enumerate(cur_det)]
        for (i, j) in pairs:
            prev_tublets[c][i].update(cur_det[j, 4], j, cur_det[j, 5:])
            cur_tublet[j] = prev_tublets[c][i]
            cur_dets[c][j, 4] = cur_tublet[j].assign_new_last()
        cur_tublets.append(cur_tublet)
    return cur_dets, cur_tublets


def make_result(bboxes, labels, embed, num_classes):
    embed_dim = embed.size(1)
    if bboxes.shape[0] == 0:
        return [
            torch.zeros((0, 5 + embed_dim), dtype=torch.float32)
            for _ in range(num_classes - 1)
        ]
    else:
        ret = []
        for i in range(num_classes - 1):
            ind = (labels == i)
            cls_det = torch.cat(
                [bboxes[ind, :], embed[ind, :]], dim=-1)
            ret.append(cls_det)
        return ret


def extract_result(results):
    return [r[:, :5].cpu().numpy() for r in results]


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

        self.prev_det = None
        self.tublets = None

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

    # def simple_test(self, img, img_meta, rescale=False):
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #     bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels, _ in bbox_list
    #     ]
    #     return bbox_results[0]

    def simple_test(self, img, img_meta, rescale=False):
        is_first = img_meta[0]['is_first']

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        bboxes, labels, embeds = bbox_list[0]

        cur_dets = make_result(bboxes, labels, embeds, self.bbox_head.num_classes)
        if is_first:
            self.prev_det, self.tublets = update_tublets(None, cur_dets, None)
            return extract_result(cur_dets)
        else:
            self.prev_det, self.tublets = update_tublets(
                self.prev_det, cur_dets, self.tublets)
            return extract_result(self.prev_det)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
