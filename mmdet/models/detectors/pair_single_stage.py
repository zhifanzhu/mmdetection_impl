import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .pair_base import PairBaseDetector

import numpy as np
import torch
from mmdet.core.bbox import bbox_overlaps
import cv2


class ScoredPath(object):
    def __init__(self, score, ind):
        self.score_list = [score]
        self.ind_list = [ind]
        self.cnt = 0
        self.score = score

    def update(self, score, ind):
        """ Update track as well as current vec"""
        self.score_list.append(score)
        self.ind_list.append(ind)
        self.score += score
        self.cnt += 1

    def __len__(self):
        return len(self.ind_list)

    def __repr__(self):
        return 'ScoredPath: ' + repr(self.ind_list)

    def assign_new_last(self):
        return torch.mean(torch.Tensor(self.score_list))


def get_p(bboxes):
    """ bboxes: [N, 5] """
    bboxes = bboxes.numpy()
    xs = (bboxes[:, 2] + bboxes[:, 0]) / 2
    ys = (bboxes[:, 3] + bboxes[:, 1]) / 2
    return np.stack([xs, ys], axis=1).reshape(-1, 1, 2)


def retrieve_box(bboxes, p):
    bboxes = bboxes.numpy()
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    p = p.reshape(-1, 2)
    x1 = p[:, 0] - w / 2
    x2 = p[:, 0] + w / 2
    y1 = p[:, 1] - h / 2
    y2 = p[:, 1] + h / 2
    return np.stack([x1, y1, x2, y2, bboxes[:, -1]], axis=1).reshape(-1, 5)

# def create_matrix(I, J):
#     """
#
#     Args:
#         I: [N, 4 + 1], Tensor
#         J: [M, 4 + 1], Tensor
#     Returns:
#         [N, M]
#     """
#     iou = bbox_overlaps(I[:, :4], J[:, :4])
#     mat = 1. / (prod * iou)
#     return mat

def match_and_pairs(prev, cur):
    num_cols = len(cur)
    if len(prev) == 0:
        return []
    if len(cur) == 0:
        return []
    # dist_mat = create_matrix(prev, cur)
    dist_mat = 1.0 / bbox_overlaps(
        cur.new_tensor(prev),
        cur)
    pairs = []
    while not torch.isinf(dist_mat.min()):
        min_ind = dist_mat.argmin()
        i, j = min_ind // num_cols, min_ind % num_cols
        pairs.append((i, j))
        dist_mat[i, :] = float('inf')
        dist_mat[:, j] = float('inf')
    return pairs


def update_tublets(est_dets, cur_dets, prev_tublets):
    if prev_tublets is None and est_dets is None:
        cur_tublets = []
        for c in range(30):
            cur_tublet = [ScoredPath(
                d[4], i) for i, d in enumerate(cur_dets[c])]
            cur_tublets.append(cur_tublet)
        return cur_dets, cur_tublets

    cur_tublets = []
    for c in range(30):
        prev_det = est_dets[c]
        cur_det = cur_dets[c]
        pairs = match_and_pairs(prev_det[:, :4],
                                cur_det[:, :4])

        cur_tublet = [ScoredPath(
            d[4], i) for i, d in enumerate(cur_det)]
        for (i, j) in pairs:
            prev_tublets[c][i].update(cur_det[j, 4], j)
            cur_tublet[j] = prev_tublets[c][i]
            cur_dets[c][j, 4] = cur_tublet[j].assign_new_last()
        cur_tublets.append(cur_tublet)
    return cur_dets, cur_tublets


def make_result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [
            torch.zeros((0, 5), dtype=torch.float32)
            for _ in range(num_classes - 1)
        ]
    else:
        ret = []
        for i in range(num_classes - 1):
            ind = (labels == i)
            cls_det = bboxes[ind, :]
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
        self.lk_params = dict(winSize=(32, 32), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_memory = None
        self.prev_gray = None
        self.prev_bboxes = None
        self.prev_labels = None

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

        cur_dets = make_result(bboxes, labels, self.bbox_head.num_classes)
        if is_first:
            cur_dets, self.tublets = update_tublets(None, cur_dets, None)
            self.prev_gray = img_meta[0]['raw_gray']
            self.prev_bboxes = bboxes
            self.prev_labels = labels
            return extract_result(cur_dets)
        else:
            # 1 transform by lk flow
            prev_bboxes_cpu = self.prev_bboxes.cpu()
            p0 = get_p(prev_bboxes_cpu)
            cur_gray = img_meta[0]['raw_gray']
            p_est, _, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, cur_gray, p0, None, **self.lk_params)
            bboxes_est = retrieve_box(prev_bboxes_cpu, p_est)
            est_dets = make_result(
                torch.from_numpy(bboxes_est), self.prev_labels, self.bbox_head.num_classes)

            # 2 update
            cur_dets, self.tublets = update_tublets(
                est_dets, cur_dets, self.tublets)
            self.prev_gray = img_meta[0]['raw_gray']
            self.prev_bboxes = bboxes
            self.prev_labels = labels
            return extract_result(cur_dets)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
