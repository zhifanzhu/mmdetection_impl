import os
import os.path as osp

import numpy as np
import mmcv

from visdrone.utils import result_utils
from visdrone.utils import test_augs
from mmdet import ops


def read_txtdir(in_dir, nms_param=None, num_classes=10):
    """
    naming:
         fstem__H_W_x_y_w_h.jpg,
    :param in_dir:
    :param nms_param: dict
    :param num_classes: predefined
    :return: Dict from fstem to its merged_results,
        which is list(stem idx) of list(class) of [N, 5]
    """
    patches_all = os.listdir(in_dir)
    # mmcv.mkdir_or_exists(out_dir)
    dets_out = dict()
    for fname in patches_all:
        fp = osp.join(in_dir, fname)
        # split fname
        fstem, info = fname.replace('.txt', '').split('__')
        H, W, x, y, w, h = np.asarray(info.split('_'), np.int32)
        ##
        # if int(x) != W - w - 1 or int(y) != H - h - 1:
        #     continue
        ##
        if fstem not in dets_out:
            dets_out[fstem] = [[] for _ in range(num_classes)]

        with open(fp) as fid:
            dets = result_utils.single_txt2det(fid)
            for c, c_det in enumerate(dets):
                dets_tmp = result_utils._rearrange_bbox(c_det, x, y)
                # dets_tmp = test_augs.bbox_score_revert_crop(c_det, (x, y))
                dets_out[fstem][c].append(dets_tmp)

    for fstem, dets in dets_out.items():
        dets_out[fstem] = result_utils.concat_01n(dets_out[fstem], nms_param)
    return dets_out


def read_origin(in_dir):
    """
        Returns: dict
    """
    ori_all = os.listdir(in_dir)
    # mmcv.mkdir_or_exists(out_dir)
    dets_out = dict()
    for fname in ori_all:
        fp = osp.join(in_dir, fname)
        # split fname
        fstem = fname.replace('.txt', '')
        with open(fp) as fid:
            dets = result_utils.single_txt2det(fid)
            if fstem not in dets_out:
                dets_out[fstem] = dets
    return dets_out


def dict2txt(det_dict, out_dir):
    mmcv.mkdir_or_exist(osp.expanduser(out_dir))
    for key, dets in det_dict.items():
        txt_name = osp.join(out_dir, key + '.txt')
        result_utils.single_det2txt(dets, txt_name)


def read_txtdir_with_output(in_dir, out_dir,  nms_param=None, num_classes=10):
    det_dict = read_txtdir(in_dir, nms_param=nms_param, num_classes=num_classes)
    dict2txt(det_dict, out_dir)
    # mmcv.mkdir_or_exist(osp.expanduser(out_dir))
    # for key, dets in det_dict.items():
    #     txt_name = osp.join(out_dir, key + '.txt')
    #     result_utils.single_det2txt(dets, txt_name)


def merge_dicts(dicts, nms_param=None):
    """
        Given a list of  dict of results, return single dict,
        All dict have the same keys.
    """
    num_classes = 10
    ret = dict()
    for d in dicts:
        for fname, dets in d.items():
            if fname not in ret:
                ret[fname] = [[] for _ in range(num_classes)]
            for c, c_det in enumerate(dets):
                ret[fname][c].append(c_det)
    for fname in ret.keys():
        ret[fname] = result_utils.concat_01n(ret[fname], nms_param)
    return ret
