import torch

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .seq_base import SeqBaseDetector
from mmcv.runner.checkpoint import load_checkpoint

""" Sequantial modeling of objects.

New Files:
[1] mmdet/models/anchor_heads/rich_retina_head.py 
[2] mmdet/models/bbox_heads/tx_head.py 
[3] mmdet/models/detectors/seq_tx.py 

Design:
1. Batch is 1
2. TWO heads?
3. Currenty, use topk for good box, no threshold used

Documents:
1. Rescale set to True or False, may cause different iou, hence lead to 
    different # of boxes.
2. Didn't write batching since it need to handling padding.
3. Pseudo Sampler use all available boxes

First try:
1.不加pos_enc
2.用MultiHeadAtt
4.Dataset分两步: i)先用val, 然后iden, 看test结果; ii) 再val过拟合 iii)再train

Transformer's Dataset:
1. Input always consecutive (1,2,3,4), no jumps.

TODO
1. Write Test time: 只取Score/或者前4个 (skip connection?)
2. Run with no modification, see test results.
2. Last frame use N, should use ALL? if fast, use ALL
4. RoIAlign to RoIPool?

Hyper parameter tuning:
1. MHA vs Transformer
2. skip vs no skip
3. seq length
"""


def do_rescale(bboxes_with_scores, img_meta):
    scale_factor = img_meta[0]['scale_factor']
    if isinstance(scale_factor, float):
        bboxes_with_scores[:, :4] /= scale_factor
    else:
        bboxes_with_scores[:, :4] /= torch.from_numpy(
            scale_factor).to(bboxes_with_scores.device)
    return bboxes_with_scores


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

        self.test_seq_len = 4
        self.memory = []

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x


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

        losses = self.tx_head.forward_loss(
            x, bbox_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, self.train_cfg)
        return losses

    def temporal_test(self, img, img_meta, seq_len, rescale=False):
        raise NotImplementedError

    def simple_test(self, img, img_meta, rescale=False):
        frame_ind = img_meta[0]['frame_ind']
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        rescale = False
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        det_bboxes, det_labels, feat = self.bbox_head.get_bboxes(*bbox_inputs)[0]

        if frame_ind < self.test_seq_len - 1:
            tgt_feat, _ = self.tx_head.get_src_feat(
                x, [(det_bboxes, det_labels, feat)])
            self.memory.append(tgt_feat)
            # Original `bbox_results` set rescale to True
            det_bboxes = do_rescale(det_bboxes, img_meta)
            single_bbox_results = bbox2result(
                det_bboxes, det_labels, self.bbox_head.num_classes)
        else:
            tgt_feat, ind_list = self.tx_head.get_src_feat(
                x, [(det_bboxes, det_labels, feat)])
            self.memory.append(tgt_feat)
            src_feat = torch.cat(self.memory)
            tgt_out, _ = self.tx_head.seq_model(tgt_feat, src_feat, src_feat)
            cur_score_feat = self.tx_head.remap_scores(feat, ind_list[0], tgt_out)

            # Again, final bboxes need to be rescaled
            det_bboxes_pre = do_rescale(det_bboxes, img_meta)
            det_bboxes, det_labels = self.tx_head.get_bboxes(
                cur_score_feat, det_bboxes_pre, self.test_cfg)
            single_bbox_results = bbox2result(
                det_bboxes, det_labels, self.tx_head.num_classes)
            self.memory.pop(0)

        return single_bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
