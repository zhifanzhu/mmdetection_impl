import numpy as np

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet import ops
from time import time

nms_func = ops.nms

MAX_BBOXES = 200


def seq_nms():
    pass


def seq_nms_core(proposals, num_frames, num_bboxes_per_frame, nms_thr=0.5, link_thr=0.5, rescore='avg'):
    """ Given a array of [F, N, 5], where F denote number of frames,
        N denote number of bboxes in one frame,
        output suppressed bboxes by frames.
    :param proposals: [F, N, 5] np array.
    :param num_frames: int
    :param num_bboxes_per_frame: list of int
    :param nms_thr: float
    :param link_thr: float
    :param rescore: rescore method
    :return: list(frame) of [N, 5]
    """
    proposals = np.copy(proposals)
    result = [[] for _ in range(num_frames)]
    valid_proposals = [list(range(v)) for v in num_bboxes_per_frame]
    num_valid_proposals = sum(num_bboxes_per_frame)

    while num_valid_proposals != 0:
        print(num_valid_proposals)
        best_seq_ind, best_seq, begin_t, end_t = find_best_seq(
            proposals, num_frames, thr=link_thr, rescore=rescore)
        if len(best_seq) == 1 and best_seq[0][-1] == 0:
            break
        for t in range(begin_t, end_t + 1):
            sel_box1 = np.copy(best_seq[t - begin_t])
            result[t].append(sel_box1)
            # get box2 left
            box2 = proposals[t][valid_proposals[t]]
            iou = bbox_overlaps(sel_box1[None, :], box2).squeeze()
            sup_inds = (iou > nms_thr).nonzero()[0].tolist()
            if len(sup_inds) != 0:
                _i = 0
                valid_p_buf = np.copy(valid_proposals[t])
                for ind, p in enumerate(valid_p_buf):
                    if ind == sup_inds[_i]:
                        proposals[t, p, :] = 0.0
                        valid_proposals[t].remove(p)
                        _i += 1
                        if _i >= len(sup_inds):
                            break
                num_valid_proposals -= len(sup_inds)

    for i, res in enumerate(result):
        if len(res) > 0:
            result[i] = np.stack(res, 0)
    return result


def find_best_seq(proposals, num_frames, thr=0.5, rescore='avg'):
    """ Find max seq(trajectory) that maximize the score:
        best_seq = argmax_{i_s, i_{s+1}, ..., i_e} \sum_{t=s}^{e} score(proposals[t][i_t])
        s.t. 0 <= s <= e < len(proposals)
        s.t IoU(proposals[t][i_t], proposals[t][i_{t+1}]) > seq_thresh for any s <= t < e.

        Before input, user should pad proposals to have Fixes shape.
    :param proposals: [F, MAX_BBOXES, 5] np array.
    :param num_frames: int
    :param thr: specify the link threshold
    :param rescore: rescore method during rescoring, 'avg' or 'max'
    :return: tuple of ([F, ], [F, 1, 5], begin_frame, end_frame),
        first is best index in each frame, second is bbox in each frame.
        begein_frame and end_frame are inclusive.
    """
    assert rescore in ('avg', 'max')
    # first element store current score, second stores prev index.
    cum_score = np.zeros([num_frames, MAX_BBOXES, 2], np.float32) - 1  # [F, N, 2]
    cum_score[0, :, 0] = proposals[0, :, -1]

    for t in range(1, num_frames):
        cur_bb = proposals[t][:, :-1]  # [N, 4]
        cur_sc = proposals[t][:, -1]  # [N,]
        prev_bb = proposals[t-1][:, :-1]
        prev_sc = cum_score[t-1, :, 0]
        iou = bbox_overlaps(cur_bb, prev_bb, mode='iou')
        score = cur_sc[:, None] + np.tile(prev_sc, [MAX_BBOXES, 1]) * (iou > thr)  # [N,1] + [N,N]
        cum_score[t, :, 1] = score.argmax(-1)
        cum_score[t, :, 0] = score.max(-1)
        invalid = (np.cumsum(iou > thr, 1)[:, -1] == 0)
        cum_score[t, invalid, 1] = -1  # set backward ind with no prev overlap to negative

    # highest score may not occur at last frame since some link might break in middle
    max_per_frame = cum_score[:, :, 0].max(1)  # [F,]
    T_end = max_per_frame.argmax()
    end_frame = T_end
    begin_frame = T_end

    end_idx = int(cum_score[T_end, :, 0].argmax())
    best_seq_index = [end_idx]
    best_seq = [proposals[T_end, end_idx, :]]
    for t in range(T_end, 0, -1):
        end_idx = int(cum_score[t, end_idx, 1])
        if end_idx == -1:
            begin_frame = t
            break
        best_seq.insert(0, proposals[t - 1, end_idx, :])
        best_seq_index.insert(0, end_idx)
        begin_frame -= 1

    # Do rescore
    best_seq = np.stack(best_seq, 0)
    if rescore == 'avg':
        best_seq[:, -1] = np.mean(best_seq[:, -1])
    else:
        best_seq[:, -1] = np.max(best_seq[:, -1])
    return best_seq_index, best_seq, begin_frame, end_frame

