import argparse
import os

import mmcv
import torch
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector

from tools import test as mmtest

from visdrone.utils import result_utils


"""
This script have 4 phases:
1. generate multiple scale results by modifying cfg., results form a list
2. merge multiscale result
3a. run coco eval (optional)
4. convert results to txt and save.

Possibly, the test dataset is already patches, then we could do:
1 & 2. same as above, note multiscale result for many patches
3b. merge patches
4. convert merged results to txt and save. (implicitly using result_utils.merge_patch to get stem filename)
We don't do coco eval here since we need coco img_infos(ids &c).  todo later.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--txtout', help='outputdir for txtfile')
    parser.add_argument('--patch', help='whether test on patch, see above notes.')
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
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
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
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # Phase 1. generate multiscale outputs
    # scales = ((1333, 800), (1500, 1500))  # move to args
    scales = cfg.data.test.img_scale
    if isinstance(scales, list):
        # list is for two_stage aug test
        scales = [scales]
    elif isinstance(scales, tuple):
        if isinstance(scales[0], int):
            # single scale
            scales = (scales, )
    else:
        raise ValueError('scales either list or tuple')
    outputs_list = []
    for scale in scales:
        print('\nINFO: evaluating :', scale)
        cfg.data.test.img_scale = scale
        dataset = get_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = mmtest.single_gpu_test(model, data_loader, args.show)
        else:
            model = MMDistributedDataParallel(model.cuda())
            outputs = mmtest.multi_gpu_test(model, data_loader, args.tmpdir)
        outputs_list.append(outputs)  # xyxy

    # Phase 2. merge results
    if len(scales) == 1:
        outputs = outputs_list[0]
    else:
        # list(scales) of list(img) of list(cls) of [N, 4]
        outputs = result_utils.concat_100n(outputs_list)

    # Phase 3a. eval coco
    rank, _ = get_dist_info()
    if not args.patch and args.out and rank == 0:
    # if  args.out and rank == 0:
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
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Phase 3b. merge patches
    max_det = 500
    if args.patch:
        print('\nINFO: merging patch...')
        # output is Dict
        outputs = result_utils.merge_patch(dataset, outputs, iou_thr=0.5, max_det=max_det)

        # Phase 3c. eval merge on coco
        # manipulate ann_file and img_prefix
        def _remove_extra(string):
            string = string.replace('-patch', '')
            return string.replace('-1024', '').replace('-640', '')
        cfg.data.test.ann_file = _remove_extra(cfg.data.test.ann_file)
        cfg.data.test.img_prefix = _remove_extra(cfg.data.test.img_prefix)
        dataset = get_dataset(cfg.data.test)
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
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Phase 4. generate txt
    save_dir = args.txtout
    mmcv.mkdir_or_exist(save_dir)
    if args.patch:
        result_utils.save_merge_patch_out(outputs, save_dir)
    else:
        result_utils.many_det2txt(dataset, outputs, save_dir)


if __name__ == '__main__':
    main()
