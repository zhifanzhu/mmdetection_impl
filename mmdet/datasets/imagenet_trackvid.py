import os.path as osp
import numpy as np

from .registry import DATASETS
from mmdet.datasets import PairVIDDataset


"""
At train time, prepare TWO frames input
At test time, prepare SINGLE frame input.

This fulfills the training scheme of: i) D & T  ii) RetinaTrack.
"""


@DATASETS.register_module
class TrackVIDDataset(PairVIDDataset):

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116',
               'n02958343', 'n02402425', 'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
               'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049',)
    DATASET_NAME = 'vid'

    def __init__(self,
                 **kwargs):
        super(TrackVIDDataset, self).__init__(**kwargs)

    def prepare_train_img(self, idx):
        """ Pipelines:
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotationsWithTrack', with_bbox=True, skip_img_without_anno=False),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundleWithTrack'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_trackids']),
        """
        img_info = self.img_infos[idx]
        frame_ind = img_info['frame_ind']
        foldername = img_info['foldername']
        num_frames = img_info['num_frames']
        ann_info = self.get_ann_info(idx, frame_ind)
        filename = osp.join(foldername, f"{frame_ind:06d}.JPEG")
        img_info['filename'] = filename
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        flip = results['img_meta'].data['flip']

        ref_frame_ind = max(
            min(frame_ind + np.random.randint(self.min_offset, self.max_offset+1),
                num_frames - 1), 0)
        ref_ann_info = self.get_ann_info(idx, ref_frame_ind)
        ref_filename = osp.join(foldername, f"{ref_frame_ind:06d}.JPEG")
        img_info['filename'] = ref_filename
        ref_results = dict(img_info=img_info, ann_info=ref_ann_info)

        self.pre_pipeline(ref_results)
        if self.match_flip:
            # See Parent class discussion
            ref_results['flip'] = flip
        ref_results = self.pipeline(ref_results)

        results['ref_img'] = ref_results['img']
        results['ref_img_meta'] = ref_results['img_meta']
        results['ref_bboxes'] = ref_results['gt_bboxes']
        results['ref_labels'] = ref_results['gt_labels']
        results['ref_trackids'] = ref_results['gt_trackids']

        if len(results['gt_bboxes'].data) == 0:
            return None
        return results
