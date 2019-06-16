#
# ktw361@2019.6.9
#

__author__ = 'ktw361'

import argparse
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.runner import get_dist_info

from mmdet.apis import init_detector, inference_detector
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.ops.nms import nms_wrapper

from visdrone.utils import test_augs
from visdrone.utils.output_to_txt import write_result_into_txt


def single_gpu_test(model, data_loader, show=False):
    """ infenrence_detector will take care of ImgTransforms."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for idx in range(len(dataset)):
        img_info = dataset.img_infos[idx]
        img = mmcv.imread(osp.join(dataset.img_prefix,
                                   img_info['filename']))
        # step 1. do 10 crop
        ori_shape = (img_info['height'], img_info['width'])
        crops, coors = test_augs.get10crops(img, ori_shape)

        # step 2. run detection on crops and do revertion
        # single_bboxes = []
        # single_scores
        single_results = []
        with torch.no_grad():
            for ret, coor in zip(inference_detector(model, crops), coors):
                single_result = [
                    test_augs.bbox_score_revert_crop(_ret, coor)
                    for _ret in ret
                ]
                single_results.append(single_result)

        # step 3. merge all det result as if one big result
        num_classes = len(single_results[0])
        per_cls_bboxes = [[] for _ in range(num_classes)]  # [ []*10]
        for result in single_results:
            for i, res in enumerate(result):
                per_cls_bboxes[i].append(res)
        for i in range(num_classes):
            per_cls_bboxes[i] = np.concatenate(per_cls_bboxes[i], 0)

        # step 4. post-processing result of single image.
        # TODO(ktw361)
        import functools
        nms_func = functools.partial(nms_wrapper.nms)
        per_cls_bboxes = transform_results_by_nms(per_cls_bboxes,
                                                  nms_func)

        # # step 5. (opt), save per crop result?
        # img_name = img_info['filename'].split('.')[0]
        # img_name = img_name.split('/')[-1]
        # for i, sin_res in enumerate(single_results):
        #     txt_file = '{}_{}.txt'.format(img_name, i)
        #     out_dir = '/tmp/fuckyoudir'
        #     txt_file = osp.join(out_dir, txt_file)
        #     write_result_into_txt(sin_res, txt_file)

        result = per_cls_bboxes
        results.append(result)
        prog_bar.update()  # only one img

    return results


def transform_results_by_nms(res, nms_func, iou_thr = 0.5):
    for i, c_res in enumerate(res):
            bb, ind = nms_func(c_res.astype(np.float32), iou_thr)
            res[i] = bb
    return res


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg = mmcv.Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint)

    outputs = single_gpu_test(model, data_loader)

    # should get outputs from above
    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_file = args.out + '.json'
                    results2json(dataset, outputs, result_file)
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}.json'.format(name)
                        results2json(dataset, outputs_, result_file)
                        coco_eval(result_file, eval_types, dataset.coco)


if __name__ == '__main__':
    main()
