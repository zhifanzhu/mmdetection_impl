import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS

from mmdet.models.roi_extractors import SingleRoIExtractor
from mmdet.core import bbox2roi, build_assigner
from mmdet.core.bbox import PseudoSampler


def get_top_n_boxes(box_list, num):
    """
    Args:
        box_list: ([N_i,5],[N_,],[N, C]) x T*1
        num: number of box for each in T*1

    Returns:
        1. [[N_j, 5] x T*1]
        2. [[N_j, C] x T*1]
        3. [[N_j,] x T*1]
         where N_j <= num
    """
    good_box_list = []
    feat_seq = []
    ind_list = []
    for boxes, labels, feat in box_list:
        num_boxes = min(num, len(boxes))
        _, ind = boxes[:, -1].topk(num_boxes)
        good_box_list.append(boxes[ind])
        feat_seq.append(feat[ind])
        ind_list.append(ind)
    feat_seq = torch.cat(feat_seq, 0)

    return good_box_list, feat_seq, ind_list


@HEADS.register_module
class TxHead(nn.Module):

    def __init__(self,
                 num_classes=31,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(TxHead, self).__init__()
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.fp16_enabled = False

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.feat_channels = 256
        self.num_box_per_frame  = 64

        self.seq_channels = self.feat_channels + (self.num_classes - 1) + 4

        self.seq_model = nn.MultiheadAttention(self.seq_channels, 2)

        self.roi_extractor = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=self.feat_channels,
            featmap_strides=[8, 16, 32, 64])

        self.debug_imgs = None

    def init_weights(self):
        # TODO init weight
        pass

    def get_seq_feat(self, fpn, box_box_list, feat_seq):
        """

        Args:
            fpn: [ [T*1, C, H, W] * 5]
            box_box_list: [N_i,5] x T*B
            feat_seq: [T*1, N_i, C]

        Returns:
            1: [T*M*1, 4 + 30 + 256]
            where M = self.num_box_per_frame,
            select top M for each of T*1
            if N < M, pad with zeros
        """
        rois = bbox2roi(box_box_list)
        roi_feat = self.roi_extractor(fpn, rois)
        roi_feat = nn.functional.avg_pool2d(
            roi_feat, (7, 7)).view(-1, self.feat_channels)

        loc_vec = torch.cat(
            [b[:, :4] for b in box_box_list], dim=0)
        src_feat = torch.cat(
            [feat_seq, loc_vec], dim=-1)
        src_feat = torch.cat([src_feat, roi_feat], dim=-1)

        return src_feat

    def get_src_feat(self, fpn, box_list):
        """

        Args:
            fpn: [ [T*1, C, H, W] * 5]
            box_list: ([N_i,5],[N_,], [N, C]) x T*1

        Returns:
            1: [T*N_j, 1, 4 + 30 + 256]
            2: [[N_i, ] x T]
            where N_j <= self.num_box_per_frame,
            select top M for each of T*1
        """
        good_box_list, feat_seq, ind_list = get_top_n_boxes(
            box_list, self.num_box_per_frame)
        src_feat = self.get_seq_feat(fpn, good_box_list, feat_seq)
        src_feat = src_feat.view(-1, 1, self.seq_channels)

        return src_feat, ind_list

    @staticmethod
    def remap_scores(cur_feats, last_box_inds, target_out):
        # TODO use skip?
        # cur_feats[last_box_inds, :] += target_out[:, 0, 4:4+30]
        cur_feats[last_box_inds, :] = target_out[:, 0, 4:4+30]
        return cur_feats

    def forward_loss(self,
                     fpn,
                     bbox_list,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore,
                     train_cfg):
        cur_bbox, cur_labels, cur_score_vec = bbox_list[-1]

        #####
        # Transformer!
        #####
        src_feat, ind_list = self.get_src_feat(fpn, bbox_list)  # [T*N_i, 1, 256]
        num_tgt = len(ind_list[-1])
        tgt_feat = src_feat[-num_tgt:]

        tgt_out, _ = self.seq_model(tgt_feat, src_feat, src_feat)  # [N, 1, 256]
        cur_score_vec = self.remap_scores(cur_score_vec, ind_list[-1], tgt_out)

        #######################
        # Loss Part
        #######################
        gt_bboxes = gt_bboxes[-1]
        gt_labels = gt_labels[-1]
        gt_bboxes_ignore = gt_bboxes_ignore[-1] if gt_bboxes_ignore is not None else None
        bbox_assigner = build_assigner(train_cfg.assigner)
        # TODO inspect sampler

        assign_result = bbox_assigner.assign(cur_bbox,
                                             gt_bboxes,
                                             gt_bboxes_ignore,
                                             gt_labels)
        sampling_result = PseudoSampler().sample(
            assign_result, cur_bbox[:, :4], gt_bboxes)
        # Sample according to sampling result
        cur_score_vec = torch.cat([
            cur_score_vec[sampling_result.pos_inds],
            cur_score_vec[sampling_result.neg_inds]], dim=0)

        losses = dict()
        bbox_targets = self.get_target(
            [sampling_result], gt_bboxes, gt_labels, train_cfg)
        loss_score = self.loss(cur_score_vec, None,
                                       *bbox_targets)
        losses.update(loss_score)
        #######################
        # End of loss Part
        #######################
        return losses

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', ))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   cls_score,
                   bboxes_final,
                   cfg):
        """ No need for delta2bbox, nms.
        Copied from multi_nms
        """
        assert len(cls_score) == len(bboxes_final)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)

        num_classes = scores.shape[1]
        bboxes, labels = [], []
        for i in range(1, num_classes):
            cls_inds = scores[:, i] > cfg.score_thr
            if not cls_inds.any():
                continue
            # get bboxes and scores of this class
            _scores = cls_score[cls_inds, i]
            _bboxes = bboxes_final[cls_inds, :4]  # modified
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_labels = _bboxes.new_full((cls_dets.shape[0], ),
                                          i - 1,
                                          dtype=torch.long)
            bboxes.append(cls_dets)
            labels.append(cls_labels)
        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
        else:
            bboxes = bboxes_final.new_zeros((0, 5))
            labels = bboxes_final.new_zeros((0, ), dtype=torch.long)

        return bboxes, labels

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
