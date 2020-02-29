import os
import os.path as osp

import json
import time
import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class AntiUavDataset(Dataset):
    CLASSES = ('Drone', )

    def __init__(self,
                 ann_file,
                 pipeline,
                 num_frames_per_clip=-1,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        begin_time = time.time()
        self.img_infos = self.load_annotations(self.ann_file, num_frames_per_clip)
        print('load_annotations time: {:.1f}s from {}'
              .format(time.time() - begin_time, ann_file))
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file, num_frames_per_clip):
        """ ann_file is train.txt """
        img_infos = []
        with open(ann_file) as fp:
            videos = fp.readlines()
            videos = [v.strip() for v in videos]

        # ALL same width/heights
        rgb_width = 1920
        rgb_height = 1080
        ir_width = 640
        ir_height = 512
        for video in videos:
            video_root = osp.join(self.img_prefix,
                                  video)
            rgb_root = osp.join(video_root, 'RGB')
            num_frames = len(os.listdir(rgb_root))
            if num_frames_per_clip < 0:
                step = 1
            else:
                step = num_frames // num_frames_per_clip
            for i in range(0, num_frames, step):
                # img_id : img_prefix/2019xxx/RGB/0001
                img_id = '/'.join([rgb_root, f"{i:06d}"])
                filename = img_id + '.jpg'
                img_infos.append(
                    dict(frame_id=i, vid_root=video_root,
                         filename=filename, width=rgb_width,
                         height=rgb_height))
        return img_infos


    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        vid_root = self.img_infos[idx]['vid_root']
        frame_id = self.img_infos[idx]['frame_id']
        rgb_json_path = osp.join(vid_root, 'RGB_label.json')
        with open(rgb_json_path) as fp:
            rgb_data = json.load(fp)
        rgb_boxes = rgb_data['gt_rect']

        # Only one object, i.e. tracking dataset
        box = rgb_boxes[frame_id]
        xmin, ymin, box_w, box_h = box
        labels = ['Drone', ]
        bboxes = [[xmin, ymin, xmin + box_w, ymin + box_h]]
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=np.zeros((0, 4)).astype(np.float32),
            labels_ignore=np.zeros((0, )).astype(np.int64))
        return ann

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
