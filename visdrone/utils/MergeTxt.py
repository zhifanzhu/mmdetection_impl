import os
import os.path as osp

import mmcv

from visdrone.utils import result_utils


def read_txtdir(in_dir, nms_param=None, num_classes=10):
    """
    naming:
         fstem__H_W_x_y_w_h.jpg,
    :param in_dir:
    :param nms_param: dict
    :param num_classes: predefined
    :return:
    """
    patches_all = os.listdir(in_dir)
    # mmcv.mkdir_or_exists(out_dir)
    dets_out = dict()
    for fname in patches_all:
        fp = osp.join(in_dir, fname)
        # split fname
        fstem, info = fname.replace('.txt', '').split('__')
        H, W, x, y, w, h = info.split('_')
        if fstem not in dets_out:
            dets_out[fstem] = [[] for _ in range(num_classes)]

        with open(fp) as fid:
            dets = result_utils.single_txt2det(fid)
            for c, c_det in enumerate(dets):
                dets_tmp = result_utils._rearrange_bbox(c_det, int(x), int(y))
                dets_out[fstem][c].append(dets_tmp)

    if nms_param is None:
        nms_param = dict(iou_thr=0.7, max_det=500)
    for fstem in dets_out.keys():
        dets_out[fstem] = result_utils.concat_01n(dets_out[fstem], nms_param)
    return dets_out


def read_txtdir_with_output(in_dir, out_dir,  nms_param=None, num_classes=10):
    det_dict = read_txtdir(in_dir, nms_param=nms_param, num_classes=num_classes)
    mmcv.mkdir_or_exist(osp.expanduser(out_dir))
    for key, dets in det_dict.items():
        txt_name = osp.join(out_dir, key + '.txt')
        result_utils.single_det2txt(dets, txt_name)
