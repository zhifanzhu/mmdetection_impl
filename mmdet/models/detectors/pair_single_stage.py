import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .pair_base import PairBaseDetector

import torch
from mmdet.core.bbox import bbox_overlaps
from mmdet.models.roi_extractors import SingleRoIExtractor
import torch.nn.functional as F


class ScoredPath(object):
    def __init__(self, score, ind):
        self.score_list = [score]
        self.ind_list = [ind]
        self.cnt = 0

    def update(self, score, ind, vec_cur):
        """ Update track as well as current vec"""
        self.score_list.append(score)
        self.ind_list.append(ind)
        vec_cur = (vec_cur + self.vec_prev * self.cnt) / (1 + self.cnt)
        self.cnt += 1
        return vec_cur

    def __len__(self):
        return len(self.ind_list)

    def __repr__(self):
        return 'ScoredPath: ' + repr(self.ind_list)

    def assign_new_last(self):
        return torch.max(torch.Tensor(self.score_list))


def get_roi_feat(fpn, boxes, extractor):
    rois = boxes.new_zeros(boxes.size(0), 5)
    rois[:, 1:] = boxes
    roi_feat = extractor(fpn[:4], rois.cuda())
    roi_feat = F.avg_pool2d(roi_feat, (7, 7)).view(-1, 256)
    return roi_feat


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
        self.roi_extractor = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64])

    def init_weights(self, pretrained=None):
        super(PairSingleStageDetector, self).init_weights(pretrained)
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
        x_ref = self.extract_feat(ref_img)
        x = self.pair_module(x, x_ref, is_train=True)
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
    #         x = self.pair_module(x, self.prev_memory, is_train=False)
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

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        x_cache = x
        is_first = img_meta[0]['is_first']
        if is_first:
            x = x
        else:
            x = self.pair_module(x, self.prev_memory, is_train=False)

        if self.with_neck and not self.neck_first:
            x = self.neck(x)

        # self.prev_memory = x
        self.prev_memory = x_cache

        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        bboxes, labels = bbox_list[0]
        bboxes_roi = bboxes[:, :4] / bboxes.new_tensor(img_meta[0]['scale_factor'])
        roi_feat = get_roi_feat(x, bboxes_roi, self.roi_extractor)

        cur_dets = make_result(bboxes, labels, roi_feat, self.bbox_head.num_classes)
        if is_first:
            self.prev_det, self.tublets = update_tublets(None, cur_dets, None)
            return extract_result(cur_dets)
        else:
            self.prev_det, self.tublets = update_tublets(
                self.prev_det, cur_dets, self.tublets)
            return extract_result(self.prev_det)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
