import torch
import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .seq_base import SeqBaseDetector
from mmcv.runner.checkpoint import load_checkpoint
from mmdet.models.roi_extractors import SingleRoIExtractor
from mmdet.core import bbox2roi, build_assigner
from mmdet.core.bbox import RandomSampler, PseudoSampler

""" Sequantial modeling of objects.

New Files:
[1] mmdet/models/anchor_heads/rich_retina_head.py 
[2] mmdet/models/bbox_heads/tx_head.py 
[3] mmdet/models/detectors/seq_tx.py 

Design:
1. Batch is 1
2. TWO heads?
3. Currenty, use topk for good box, no threshold used
4. Loss need ...
1. Random Sampler, random sampler should be sufficient? 100 < 256

Documents:
1. Rescale set to True or False, may cause different iou, hence lead to 
    different # of boxes.

First try:
1.不加pos_enc
2.用MultiHeadAtt
4.Dataset分两步: i)先用val, 然后iden, 看test结果; ii) 再val过拟合 iii)再train

TODO
1. WRite Test time: 只取Score/或者前4个 (skip connection?)
2. Run with no modification, see test results.
2. Last frame use N, should use ALL? if fast, use ALL
4. RoIAlign to RoIPool?

Transformer's Dataset:
1. Input always consecutive (1,2,3,4), no jumps.
"""


# def one_hotize(boxes, labels):
#     """
#     Args:
#         boxes: [N,5]
#         labels: [N]
#
#     Returns:
#         [N, 4 + 30] where N_j <= num
#     """
#     one_hot = boxes.new_zeros(len(boxes), 30)
#     one_hot[:, labels] = boxes[:, -1]
#     return one_hot

def do_rescale(bboxes_with_scores, img_meta):
    scale_factor = img_meta[0]['scale_factor']
    if isinstance(scale_factor, float):
        bboxes_with_scores[:, :4] /= scale_factor
    else:
        bboxes_with_scores[:, :4] /= torch.from_numpy(
            scale_factor).to(bboxes_with_scores.device)
    return bboxes_with_scores


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


@DETECTORS.register_module
class SeqTxSingleStage(SeqBaseDetector):

    def __init__(self,
                 det_load_from,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 tx_head=None,
                 temporal_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SeqTxSingleStage, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        assert temporal_module is None
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.init_weights(pretrained=pretrained)
        load_checkpoint(
            self, det_load_from, map_location='cpu', strict=False, logger=None)
        self.backbone.eval()
        self.neck.eval()
        self.bbox_head.eval()

        self.tx_head = builder.build_head(tx_head)
        self.num_box_per_frame = 64
        self.feat_channels = 256
        self.seq_channels = self.feat_channels + (self.bbox_head.num_classes - 1) + 4
        self.seq_model = nn.MultiheadAttention(self.seq_channels, 2)
        self.roi_extractor = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=self.feat_channels,
            featmap_strides=[8, 16, 32, 64])

        self.test_seq_len = 4
        self.memory = []

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def get_seq_feat(self, fpn, box_box_list, feat_seq):
        """

        Args:
            fpn: [ [T*B, C, H, W] * 5]
            box_box_list: [N_i,5] x T*B
            feat_seq: [T*B, N_i, C]

        Returns:
            1: [T*M*1, 4 + 30 + 256]
            where M = self.num_box_per_frame,
            select top M for each of T*B
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
            fpn: [ [T*B, C, H, W] * 5]
            box_list: ([N_i,5],[N_,], [N, C]) x T*B

        Returns:
            1: [T*N_j, 1, 4 + 30 + 256]
            2: [[N_i, ] x T]
            where N_j <= self.num_box_per_frame,
            select top M for each of T*B
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

    def forward_train(self,
                      img,
                      img_metas,
                      seq_len,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        batch = img.size(0) // seq_len
        assert batch == 1
        with torch.no_grad():
            x = self.extract_feat(img)  # [[T*1, c1, h1, w1]*5]

            outs = self.bbox_head(x)
            rescale = False
            bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        cur_bbox, cur_labels, cur_score_vec = bbox_list[-1]

        #####
        # Transformer!
        #####
        src_feat, ind_list = self.get_src_feat(x, bbox_list)  # [T*N_i, 1, 256]
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
        bbox_assigner = build_assigner(self.train_cfg.assigner)
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
        bbox_targets = self.tx_head.get_target(
            [sampling_result], gt_bboxes, gt_labels, self.train_cfg)
        loss_score = self.tx_head.loss(cur_score_vec, None,
                                       *bbox_targets)
        losses.update(loss_score)
        #######################
        # End of loss Part
        #######################
        return losses

    def temporal_test(self, img, img_meta, seq_len, rescale=False):
        raise NotImplementedError

    def simple_test(self, img, img_meta, in_dict=None, rescale=False):
        frame_ind = img_meta[0]['frame_ind']
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        rescale = False
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        det_bboxes, det_labels, feat = self.bbox_head.get_bboxes(*bbox_inputs)[0]

        if frame_ind < self.test_seq_len - 1:
            tgt_feat, _ = self.get_src_feat(x, [(det_bboxes, det_labels, feat)])
            self.memory.append(tgt_feat)
            # Original `bbox_results` set rescale to True
            det_bboxes = do_rescale(det_bboxes, img_meta)
            single_bbox_results = bbox2result(
                det_bboxes, det_labels, self.bbox_head.num_classes)
        else:
            tgt_feat, ind_list = self.get_src_feat(x, [(det_bboxes, det_labels, feat)])
            self.memory.append(tgt_feat)
            src_feat = torch.cat(self.memory)
            tgt_out, _ = self.seq_model(tgt_feat, src_feat, src_feat)
            cur_score_feat = self.remap_scores(feat, ind_list[0], tgt_out)

            # Again, final bboxes need to be rescaled
            final_bbox = do_rescale(det_bboxes, img_meta)
            det_bboxes, det_labels = self.tx_head.get_bboxes(
                cur_score_feat, final_bbox, self.test_cfg)
            single_bbox_results = bbox2result(
                det_bboxes, det_labels, self.tx_head.num_classes)
            self.memory.pop(0)

        return single_bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
