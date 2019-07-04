#
# ktw361@2019.5.27
#

""" Run detector and write output to txt file for VisDrone
    No test-time augmentation.
"""

__author__ = 'ktw361'

import argparse

import os
import os.path as osp
import torch
import mmcv
import tempfile
import shutil

from mmdet.apis import init_detector, inference_detector
from visdrone.utils import result_utils, MergeTxt

"""
    Usage: python output_to_txt.py --img_prefix /path/to/images \
        --out-dir /txt/save/dir \
        'config' \
        'checkpoint' \
        --type 'one of 'det' 'detpatch' 'vidpatch'

"""


def parse_args():
    parser = argparse.ArgumentParser(description='Run detector and output to txt files')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img-prefix', help='prefix of images')
    parser.add_argument('--out-dir', help='output directory holding result txt files')
    parser.add_argument('--type', type=str, nargs='?',
                        default='det',
                        choices=['det', 'detpatch', 'vidpatch'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('config: ', args.config)
    print('checkpoint: ', args.checkpoint)
    print('img_prefix: ', args.img_prefix)
    print('out_dir: ', args.out_dir)
    out_dir = args.out_dir
    if args.type != 'det':
        out_dir = tempfile.mkdtemp()
    mmcv.mkdir_or_exist(args.out_dir)
    print('saving to {}'.format(out_dir))

    cfg = mmcv.Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint)

    all_images = args.img_prefix
    assert osp.exists(all_images)
    all_images_list = os.listdir(all_images)
    prog_bar = mmcv.ProgressBar(len(all_images_list))
    for i, img_name in enumerate(all_images_list):
        img_full_name = osp.join(all_images, img_name)
        with torch.no_grad():
            result = inference_detector(model, img_full_name)
        txt_file = '{}.txt'.format(img_name.split('.')[0])
        txt_file = osp.join(out_dir, txt_file)
        result_utils.single_det2txt(result, txt_file)
        prog_bar.update()

    if args.type == 'detpatch':
        print('DET patch merging...')
        temp_out = out_dir
        out_dir = args.out_dir
        nms_param = dict(iou_thr=0.5, max_det=200, score_thr=0.1)
        merge_dict = MergeTxt.read_txtdir(temp_out, nms_param)
        shutil.rmtree(temp_out)
        mmcv.mkdir_or_exist(out_dir)
        MergeTxt.dict2txt(merge_dict, out_dir)

    if args.type == 'vidpatch':
        print('VID patch merging...')
        temp_out = out_dir
        out_dir = args.out_dir
        nms_param = dict(iou_thr=0.5, max_det=200, score_thr=0.1)
        merge_dict = MergeTxt.read_txtdir(temp_out, nms_param)
        shutil.rmtree(temp_out)
        # parse dict
        seq_dict = dict()
        for key, frame_det in merge_dict.items():
            splited = key.split('_')
            frame_ind = int(splited[-1])
            seq_name = '_'.join(splited[:-1])
            if seq_name not in seq_dict:
                seq_dict[seq_name] = {frame_ind: frame_det}
            else:
                seq_dict[seq_name][frame_ind] = frame_det

        mmcv.mkdir_or_exist(out_dir)
        MergeTxt.save_seq_results(seq_dict, save_dir=out_dir)


if __name__ == '__main__':
    main()
