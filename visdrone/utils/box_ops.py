#
# ktw361@2019.6.16
#
''' From tensorflow's object detection API'''

__author__ = 'ktw361'

import numpy as np
import torch

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet import ops

non_max_suppression = ops.nms


def refine_boxes_multi_class(pool_boxes,
                             num_classes,
                             nms_iou_thr,
                             voting_iou_thr=0.5,
                             device_id=None):
    """Refines a pool of boxes using non max suppression and box voting.

    Box refinement is done independently for each class.

    Args:
        pool_boxes: list of [N, 5] np/tensor, A collection of boxes to be refined.
            each entry of pool_boxes must have a rank 1 'scores' column, each
            entry represents one class.
        num_classes: (int scalar) Number of classes.
        nms_iou_thr: (float scalar) iou threshold for non max suppression (NMS).
        voting_iou_thr: (float scalar) iou threshold for box voting.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        Refined boxes.

    Raises:
        ValueError: if
        a) nms_iou_thresh or voting_iou_thresh is not in [0, 1].
        b) pool_boxes does not have a scores column.
    """
    if not len(pool_boxes) == num_classes:
        raise ValueError('num_classes must be equal to len(pool_boxes)')
    if not 0.0 <= nms_iou_thr <= 1.0:
        raise ValueError('nms_iou_thr must be between 0 and 1')
    if not 0.0 <= voting_iou_thr <= 1.0:
        raise ValueError('voting_iou_thr must be between 0 and 1')

    refined_boxes = []
    for i in range(num_classes):
        boxes_class = pool_boxes[i]
        refined_boxes_class, _ = refine_boxes(boxes_class, nms_iou_thr,
                                              voting_iou_thr,
                                              device_id)
        refined_boxes.append(refined_boxes_class)
    return refined_boxes


def refine_boxes(pool_boxes,
                 nms_iou_thr,
                 voting_iou_thr=0.5,
                 device_id=None):
    """Refines a pool of boxes using non max suppression and box voting.

    Args:
        pool_boxes: [N, 5] boxes, A collection of boxes to be refined. pool_boxes
            must have a 'scores' column.
        nms_iou_thr: (float scalar) iou threshold for non max suppression (NMS).
        voting_iou_thr: (float scalar) iou threshold for box voting.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        refined boxes.

    Raises:
        ValueError: if
        a) nms_iou_thresh or voting_iou_thresh is not in [0, 1].
        b) pool_boxes does not have a scores column.
    """
    if not 0.0 <= nms_iou_thr <= 1.0:
        raise ValueError('nms_iou_thr must be between 0 and 1')
    if not 0.0 <= voting_iou_thr <= 1.0:
        raise ValueError('voting_iou_thr must be between 0 and 1')
    if not pool_boxes.shape[1] == 5:
        raise ValueError('pool_boxes must have a \'scores\' column')

    sel_boxes, sel_inds = non_max_suppression(
            pool_boxes, nms_iou_thr, device_id)  # GPU or CPU
    if isinstance(pool_boxes, torch.Tensor):
        is_tensor = True
        sel_boxes_np = sel_boxes.detach().cpu().numpy()
        pool_boxes_np = pool_boxes.detach().cpu().numpy()
    else:
        is_tensor = False
        sel_boxes_np = sel_boxes
        pool_boxes_np = pool_boxes

    avg_boxes = box_voting(sel_boxes_np, pool_boxes_np, voting_iou_thr, device_id)

    if is_tensor:
        pool_device = pool_boxes.device
        return torch.from_numpy(avg_boxes).to(pool_device), \
               torch.tensor(sel_inds, dtype=torch.long).to(pool_device)
    else:
        return avg_boxes.astype(np.float32), sel_inds.astype(np.int64)


def box_voting(selected_boxes, pool_boxes, iou_thresh=0.5, device_id=None):
    """Performs box voting as described in S. Gidaris and N. Komodakis, ICCV 2015.

    Performs box voting as described in 'Object detection via a multi-region &
    semantic segmentation-aware CNN model', Gidaris and Komodakis, ICCV 2015. For
    each box 'B' in selected_boxes, we find the set 'S' of boxes in pool_boxes
    with iou overlap >= iou_thresh. The location of B is set to the weighted
    average location of boxes in S (scores are used for weighting). And the score
    of B is set to the average score of boxes in S.

    Args:
        selected_boxes: [N, 5] Boxes containing a subset of boxes in pool_boxes.
            These boxes are usually selected from pool_boxes using NMS.
        pool_boxes: [N, 5] Boxes containing a set of (possibly redundant) boxes.
        iou_thresh: (float scalar) iou threshold for matching boxes in
            selected_boxes and pool_boxes.

    Returns:
        Boxes containing averaged locations and scores for each box in
        selected_boxes.

    Raises:
        ValueError: if
            a) if iou_thresh is not in [0, 1].
            b) pool_boxes does not have a scores column.
    """
    if not type(selected_boxes) == type(pool_boxes):
        raise TypeError(
            'selected_boxes must be same type as pool_boxes')

    if isinstance(selected_boxes, torch.Tensor):
        raise NotImplementedError
        # return box_voting_tensor(selected_boxes, pool_boxes, iou_thresh)
    elif isinstance(selected_boxes, np.ndarray):
        # sel_th = torch.from_numpy(selected_boxes).type(torch.float32).to(device_id)
        # pool_th = torch.from_numpy(pool_boxes).type(torch.float32).to(device_id)
        # averaged_boxes = box_voting_tensor(sel_th, pool_th, iou_thresh)
        # return averaged_boxes.cpu().numpy()
        return box_voting_numpy(selected_boxes, pool_boxes, iou_thresh)
    else:
        raise TypeError(
            'Must be either a Tensor or numpy array, but got {}'.format(
                type(selected_boxes)))


def box_voting_numpy(selected_boxes, pool_boxes, iou_thr=0.5):
    """ Real box-voting for numpy.ndarray
    """
    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError('iou_thr must be between 0 and 1')
    if not pool_boxes.shape[1] == 5:
        raise ValueError('pool_boxes must have a \'scores\' column')

    iou_ = bbox_overlaps(selected_boxes[:, :4], pool_boxes[:, :4])
    match_indicator = (iou_ > iou_thr).astype(np.float32)
    num_matches = np.sum(match_indicator, 1)
    # TODO(kbanoop): Handle the case where some boxes in selected_boxes do not
    # match to any boxes in pool_boxes. For such boxes without any matches, we
    # should return the original boxes without voting.
    keep_ind = (num_matches == 0).nonzero()
    # match_assert = (num_matches > 0).all()
    scores = np.expand_dims(pool_boxes[:, -1], 1)
    score_assert = (scores >= 0).all()

    # if not match_assert:
    #     print('Each box in selected_boxes must match with at least one box'
    #           'in pool_boxes.')
    if not score_assert:
        print('Scores must be non negative.')
    sum_scores = np.matmul(match_indicator, scores)

    averaged_scores = np.reshape(sum_scores, [-1]) / num_matches
    averaged_scores[keep_ind] = selected_boxes[:, -1][keep_ind]

    box_locations = np.matmul(match_indicator,
                              pool_boxes[:, :4] * scores) / sum_scores
    averaged_boxes = np.copy(box_locations)
    averaged_boxes[keep_ind] = selected_boxes[:, :4][keep_ind]
    averaged_boxes = np.concatenate([averaged_boxes,
                                     averaged_scores[:, None]], 1)
    return averaged_boxes


# def box_voting_tensor(selected_boxes, pool_boxes, iou_thresh=0.5):
#     """ Real box-voting for torch.tensor
#     """
#     if not 0.0 <= iou_thresh <= 1.0:
#         raise ValueError('iou_thresh must be between 0 and 1')
#     if not pool_boxes.shape[1] == 5:
#         raise ValueError('pool_boxes must have a \'scores\' column')

#     if len(selected_boxes) == 0:
#         return selected_boxes

#     iou_ = bbox_overlaps_tensor(selected_boxes[:, :4], pool_boxes[:, :4])
#     match_indicator = (iou_ > iou_thresh).type(torch.float32)
#     num_matches = torch.sum(match_indicator, 1)
#     # keep_ind handles the case where some boxes in selected_boxes do not
#     # match to any boxes in pool_boxes. For such boxes without any matches, we
#     # should return the original boxes without voting.
#     keep_ind = (num_matches == 0).nonzero().squeeze()
#     scores = pool_boxes[:, -1][:, None]
#     score_assert = (scores >= 0).all()

#     if not score_assert:
#         print('Scores must be non negative.')
#     sum_scores = torch.matmul(match_indicator, scores)

#     averaged_scores = sum_scores.view([-1]) / num_matches
#     averaged_scores[keep_ind] = selected_boxes[:, -1][keep_ind]

#     box_locations = torch.matmul(match_indicator,
#                                  pool_boxes[:, :4] * scores) / sum_scores
#     averaged_boxes = box_locations.clone().detach()
#     averaged_boxes[keep_ind] = selected_boxes[:, :4][keep_ind]
#     averaged_boxes = torch.cat([averaged_boxes,
#                                 averaged_scores[:, None]], 1)
#     return averaged_boxes
