import _io
import numpy as np
import os.path as osp
from mmdet import ops


def single_det2txt(result, txt_file):
    """ write mmdetection result to txt_file

    xyxy -> xywh

    Args:
        result: list(cls) of np array with [N, 5] of [x1, y1, x2, y2, score]
        txt_file: str specify output filename
    """
    with open(txt_file, 'w') as fid:
        for i, res in enumerate(result):
            cat_id = i + 1
            if len(res) == 0:
                continue
            for det in res:
                x1, y1, x2, y2, sc = det
                w = x2 - x1
                h = y2 - y1
                fid.writelines('%d,%d,%d,%d,%.4f,%d,-1,-1\n' % (
                    x1, y1, w, h, sc, cat_id
                ))


def many_det2txt(dataset, results, save_dir):
    """
    :param dataset: Dataset object with img_infos, img_ids
    :param results: list(file idx) of list(class) of [N, 5]
    :return: dict from fstem to its merged_results,
        which is list(stem idx) of list(class) of [N, 5]
    """
    for info in dataset.img_infos:
        fname = info['file_name']
        f_id = info['id']
        txt_name = fname.replace('.jpg', '.txt').split('/')[-1]
        txt_name = osp.join(save_dir, txt_name)
        idx = dataset.img_ids.index(f_id)
        single_det2txt(results[idx], txt_name)


def single_txt2det(fid, num_classes=10):
    """ Returns detection result from one txt(image)
    Args:
        fid: opened file handler or a file name.
    """
    if not isinstance(fid, _io.TextIOWrapper):
        fid = open(fid, 'r')
    lines = fid.readlines()
    lines = [v.strip('\n') for v in lines]
    lines = [v.split(',') for v in lines]
    dets = [[] for _ in range(num_classes)]
    for line in lines:
        x1, y1, w, h, sc, label, trun, occ = line
        label = int(label)
        if label == 0 or label == 11:
            continue
        assert label > 0 and label < 11, 'Bad label'
        x1, y1 = int(x1), int(y1)
        w, h = int(w), int(h)
        score = float(sc)
        x2 = x1 + w
        y2 = y1 + h
        bbox = np.asarray([x1, y1, x2, y2, score], dtype=np.float32)
        dets[label - 1].append(bbox)
    for i in range(len(dets)):
        if len(dets[i]) == 0:
            dets[i] = np.empty([0, 5])
        else:
            dets[i] = np.stack(dets[i], 0)
    return dets


def single_seq2det(fid):
    """
        Returns: dict from framd_idx to list of dets (no class agnostic)
    """
    lines = fid.readlines()
    lines = [v.strip('\n') for v in lines]
    lines = [v.split(',') for v in lines]
    lines = np.asarray(lines)[:, :10].astype(np.int32)

    dets = dict()
    for line in lines:
        frame_id, track_id, x1, y1, w, h, sc, label, trun, occ = line
        frame_id = str(frame_id).zfill(7)
        label = int(label)
        w, h = int(w), int(h)
        x2 = x1 + w
        y2 = y1 + h
        bbox = np.asarray([x1, y1, x2, y2]).tolist()
        if frame_id not in dets:
            dets = [bbox]
        else:
            dets.append(bbox)

    for k, det in dets.items():
        if len(det) == 0:
            dets[k] = np.empty([0, 5])
        else:
            dets[k] = np.stack(det)
    return dets


# def pkl2txt(dataset, results, save_dir):
#     """
#     :param dataset: Dataset object with img_infos, img_ids
#     :param results: list(file idx) of list(class) of [N, 5]
#     :return: dict from fstem to its merged_results,
#         which is list(stem idx) of list(class) of [N, 5]
#     """
#     for info in dataset.img_infos:
#         fname = info['file_name']
#         f_id = info['id']
#         txt_name = fname.replace('.jpg', '.txt').split('/')[-1]
#         txt_name = osp.join(save_dir, txt_name)
#         idx = dataset.img_ids.index(f_id)
#         single_det2txt(results[idx], txt_name)


"""
    Notations: 
        0, 1 refer to list, n refer to  [N, 5] array
        0n, means list of [N,5], 0 will be kept, 1 will be concated according to n.
        concat_01n will output _0n, this normally means input is list(img) of list(patches) of [N, 5]
        concat_100n will output _00n, 
"""


def nms_0n(result, iou_thr=0.5):
    """
        nms for each class.
    :param result: list of [N, 5]
    :param iou_thr:
    :return: list of [N, 5]
    """
    for c, c_det in enumerate(result):
        bb, ind = ops.nms(c_det, iou_thr=iou_thr)
        result[c] = bb
    return result


def nms_00n(results, iou_thr=0.5):
    """
        nms for multiple image results
    :param results:  list of list of [N, 5]
    :param iou_thr:
    :return: same as results
    """
    for i, img_det in enumerate(results):
        results[i] = nms_0n(img_det, iou_thr)
    return results


def concat_1n(bboxes, nms_param):
    """
        Note, remember to revert bboxes before concat (center offset)

    :param bboxes: list(patches) of [N, 5], indicating patches bboxes
    :param nms_param: {'iou_thr', 'max_det'}
    :return: [N, 5], 'x_n'
    """
    if nms_param is None:
        return np.concatenate(bboxes, 0)
    else:
        bboxes = np.concatenate(bboxes, 0)
        bboxes, _ = ops.nms(bboxes, iou_thr=nms_param['iou_thr'])
        # bboxes, _ = ops.soft_nms(np.float32(bboxes), iou_thr=nms_param['iou_thr'], min_score=0.05)
        if 'score_thr' in nms_param:
            inds = bboxes[:, -1] > nms_param['score_thr']
            if len(inds) == 0:
                return np.empty([0, 5], np.float32)
            bboxes = bboxes[inds, :]
        if nms_param['max_det'] < 0:
            return bboxes
        else:
            max_det = nms_param['max_det']
            if bboxes.shape[0] > max_det:
                inds = np.argsort(bboxes[:, -1])[::-1]
                inds = inds[:max_det]
                bboxes = bboxes[inds, ...]
            return bboxes


def concat_01n(bboxes, nms_param):
    """
    :param bboxes: list(cls) of list(patches) of [N, 4]
    :param nms_param: {'iou_thr', 'max_det'}
    :return: list(cls) of [N, 4], 'x_0n'
    """
    ret = []
    for bb_list in bboxes:
        bb = concat_1n(bb_list, nms_param)
        ret.append(bb)
    return ret


def concat_10n(bboxes, nms_param):
    num_classes = len(bboxes[0][0])
    ret = [[] for _ in range(num_classes)]
    for _, bb in enumerate(bboxes):
        for c, bb_c in enumerate(bb):
            ret[c].append(bb_c)
    return concat_01n(ret, nms_param)


def concat_001n(bboxes, nms_param):
    """
    :param bboxes: list(img) of list(cls) of list(to_concat) of [N,4]
    :return: 00n
    """
    ret = []
    for bb in bboxes:
        ret.append(concat_01n(bb, nms_param))
    return ret


def concat_100n(bboxes, nms_param=None):
    """
        100n -> 00n
        example, scale of list of list
    :param bboxes: list(scales) of list(imgs) of list(cls) of [N, 4]
    :param nms_param: whether do nms on '1', if None, do nothing
    :return: list(imgs) of list(cls) of [N, 4]
    """
    num_imgs = len(bboxes[0])
    num_classes = len(bboxes[0][0])
    ret = [None for _ in range(num_imgs)]

    for sc, sc_dets in enumerate(bboxes):
        for img, img_dets in enumerate(sc_dets):
            ret[img] = [[] for _ in range(num_classes)]
            for c, c_dets in enumerate(img_dets):
                ret[img][c].append(c_dets)
    return concat_001n(ret, nms_param)


"""
    The follow is for use when test on patched dataset.
"""


def merge_patch(dataset, results, iou_thr, max_det):
    """
    For use in single_multiscale_test.py
        dataset.img_infos naming:
        fstem__H_W_x_y_w_h.jpg,
        Content:
        [{'license': 4,
          'file_name': 'images/9999937_00000_d_0000005__756_1344_0_0_640_640.jpg',
          'height': 641,
          'width': 641,
          'id': 64935149,
          }, ... ]

    :param dataset: Dataset object with img_infos, img_ids
    :param results: list(file/patch idx) of list(class) of [N, 5]
    :param iou_thr: iou thresh for merging patches
    :param max_det: usually 100 or 500
    :return: Dict from fstem to its merged_results,
        which is list(stem idx) of list(class) of [N, 5]
    """
    num_classes = len(dataset.cat_ids)

    def _get_HWxywh(string):
        return np.asarray(string.split('_'), np.int32)

    stem2idx = dict()
    stem2coor = dict()
    for info in dataset.img_infos:
        fname = info['file_name']
        f_id = info['id']
        fname = fname.replace('.jpg', '').split('/')[-1]
        stem, extra = fname.split('__')
        idx = dataset.img_ids.index(f_id)
        try:
            stem2idx[stem].append(idx)
            stem2coor[stem].append(_get_HWxywh(extra))
        except KeyError:
            stem2idx[stem] = [idx]
            stem2coor[stem] = [_get_HWxywh(extra)]
    fstems= stem2idx.keys()

    merged_results = dict()
    for fstem in fstems:  # for each meta file
        for idx, coor in zip(stem2idx[fstem], stem2coor[fstem]):  # for each patch
            merged_results[fstem] = [[] for _ in range(num_classes)]
            for c, bb_cls in enumerate(results[idx]):  # for each box_cls
                x, y = coor[2], coor[3]
                bb_cls = _rearrange_bbox(bb_cls, x, y)
                merged_results[fstem][c].append(bb_cls)
        # Suggest
        # merged_results[fstem] = concat01n(merged_results[fstem])
        for c in range(num_classes):
            bb_pre = np.concatenate(merged_results[fstem][c], 0)
            # NMS
            bb, _ = ops.nms(bb_pre, iou_thr=iou_thr)
            merged_results[fstem][c] = bb[:max_det]
    return merged_results


def _rearrange_bbox(bbox, x_offset, y_offset):
    bbox[:, :4] += np.tile((x_offset, y_offset), 2)
    return bbox


def save_merge_patch_out(output_dict, save_dir, ext='.txt'):
    for fstem, results in output_dict.items():
        save_file = osp.join(save_dir, fstem + ext)
        single_det2txt(results, save_file)
