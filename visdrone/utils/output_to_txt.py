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

from mmdet.apis import init_detector, inference_detector
from visdrone.utils import result_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Run detector and output to txt files')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img-prefix', help='prefix of annotations and images')
    parser.add_argument('--out-dir', help='output directory holding result txt files')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('config: ', args.config)
    print('checkpoint: ', args.checkpoint)
    print('img_prefix: ', args.img_prefix)
    print('out_dir: ', args.out_dir)
    mmcv.mkdir_or_exist(args.out_dir)

    cfg = mmcv.Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint)

    all_images = osp.join(args.img_prefix, 'images')
    all_images_list = os.listdir(all_images)
    prog_bar = mmcv.ProgressBar(len(all_images_list))
    for i, img_name in enumerate(all_images_list):
        img_full_name = osp.join(all_images, img_name)
        with torch.no_grad():
            result = inference_detector(model, img_full_name)
        txt_file = '{}.txt'.format(img_name.split('.')[0])
        txt_file = osp.join(args.out_dir, txt_file)
        result_utils.single_det2txt(result, txt_file)
        prog_bar.update()
        # if i % 100 == 0:
        #     print('{}/{}'.format(i, len(os.listdir(all_images))))


if __name__ == '__main__':
    main()
