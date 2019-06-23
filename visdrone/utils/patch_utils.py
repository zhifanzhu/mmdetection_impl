#
# ktw361@2019.6.23
#

import os.path as osp
import numpy as np

from mmdet import ops
from visdrone.utils import output_to_txt, multicrop_test


def merge_patch(dataset, results):
    """
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
    :param results: list(file idx) of list(class) of [N, 5]
    :return: dict from fstem to its merged_results,
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
        for c in range(num_classes):
            bb_pre = np.concatenate(merged_results[fstem][c], 0)
            # NMS
            bb, _ = ops.nms(bb_pre, iou_thr=0.7)
            merged_results[fstem][c] = bb[:500]
    return merged_results


def _rearrange_bbox(bbox, x_offset, y_offset):
    bbox[:, :4] += np.tile((x_offset, y_offset), 2)
    return bbox


def save_merged_det(merged_result, save_dir, ext='.txt'):
    for fstem, result in merged_result.items():
        save_txt = fstem + ext
        save_txt = osp.join(save_dir, save_txt)
        output_to_txt.write_result_into_txt(result, save_txt)