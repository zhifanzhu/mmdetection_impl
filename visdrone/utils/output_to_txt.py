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
import mmcv

from mmdet.apis import inference_detector
from mmdet.models import build_detector


def write_result_into_txt(result, txt_file):
    """ write mmdetection result to txt_file

    Args:
        result: list of np array with [N, 5] of [x1, y1, x2, y2, score]
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


def parse_args():
    parser = argparse.ArgumentParser(description='Run detector and output to txt files')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
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

    cfg = mmcv.Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    mmcv.runner.load_checkpoint(model, args.checkpoint)

    all_images = osp.join(args.img_prefix, 'images')
    for img_name in os.listdir(all_images):
        img_full_name = osp.join(all_images, img_name)
        result = inference_detector(model, img_full_name, cfg, device='cuda')
        txt_file = '{}.txt'.format(img_name.split('.')[0])
        txt_file = osp.join(args.out_dir, txt_file)
        write_result_into_txt(result, txt_file)


if __name__ == '__main__':
    main()
