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
        #
        # if not ((638 < w < 642) and (638 < h < 642)):
        #     continue
        #
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


"""
    VID
    vid result <- frame det <- patch det
1. to get vid result from frames det: 
    frames_dets2dict_then_output(),  (no NMS param)
2. to get vid result from patch det:
    read_txtdir_withoutput(), frames_dets2dict_then_output() (have NMS param)
    
"""


def assign_seq_results(dataset, results):
    """
        Content:
        [{'license': 4,
          'file_name': 'sequences/uavxxx/frame_idjpg',
          'height': 641,
          'width': 641,
          'id': 64935149,
          }, ... ]

    :param dataset: Dataset object with img_infos, img_ids
    :param results: list(frames) of list(class) of [N, 5]
    :return: Dict from seq_name to its frame results,
        which is list(frame_id) of list(class) of [N, 5]
    """
    seq_results = dict()
    for info in dataset.img_infos:
        fname = info['file_name']
        _, seq_name, frame_ind = fname.replace('.jpg', '').split('/')
        frame_ind = int(frame_ind)
        ds_id = info['id']
        idx = dataset.img_ids.index(ds_id)
        frame_dets = results[idx]
        try:
            seq_results[seq_name][frame_ind] = frame_dets
        except KeyError:
            seq_results[seq_name] = {frame_ind: frame_dets}
    return seq_results


def frames_dets2dict(in_dir):
    frames_dict = read_origin(in_dir)
    seq_dict = dict()
    for key, frame_det in frames_dict.items():
        splited = key.split('_')
        frame_ind = int(splited[-1])
        seq_name = '_'.join(splited[:-1])
        if seq_name not in seq_dict:
            seq_dict[seq_name] = {frame_ind: frame_det}
        else:
            seq_dict[seq_name][frame_ind] = frame_det
    return seq_dict


def frames_dets2dict_then_output(in_dir, save_dir, ext='.txt'):
    seq_dict = frames_dets2dict(in_dir)
    mmcv.mkdir_or_exist(save_dir)
    save_seq_results(seq_dict, save_dir, ext)


def save_seq_results(output_dict, save_dir, ext='.txt'):
    for seq_name, seq_results in output_dict.items():
        save_file = osp.join(save_dir, seq_name + ext)
        with open(save_file, 'w') as fid:
            for frame_ind, frame_det in seq_results.items():
                for i, res in enumerate(frame_det):
                    cat_id = i + 1
                    if len(res) == 0:
                        continue
                    for det in res:
                        x1, y1, x2, y2, sc = det
                        w = x2 - x1
                        h = y2 - y1
                        fid.writelines('%d,-1,%d,%d,%d,%d,%.4f,%d,-1,-1\n' % (
                            frame_ind, x1, y1, w, h, sc, cat_id
                        ))
