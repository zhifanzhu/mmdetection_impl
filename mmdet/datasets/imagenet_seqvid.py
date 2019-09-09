import os.path as osp
import xml.etree.ElementTree as ET
from pathlib import Path

import collections
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from mmcv.parallel.data_container import DataContainer

import random
import time
import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class SeqVIDDataset(Dataset):

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116',
               'n02958343', 'n02402425', 'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
               'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049',)

    def __init__(self,
                 ann_file,
                 pipeline,
                 seq_len,
                 skip=None,
                 min_size=None,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.skip = skip
        self.seq_len = seq_len

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
        self.vid_infos = self.load_annotations(self.ann_file)
        print('load_annotations time: {:.1f}s from {}'
              .format(time.time() - begin_time, ann_file))
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def __len__(self):
        return len(self.vid_infos)

    def load_annotations(self, ann_file):
        vid_infos = []
        vid_ids = mmcv.list_from_file(ann_file)

        def _train_get_vid_id(_id_line):
            _4d_8d, _start_ind, _end_ind, _num_frames = _id_line.split(' ')
            return _4d_8d, int(_start_ind), int(_end_ind), int(_num_frames)

        def _val_get_vid_id(_id_line):
            _vid_id, _start_ind, _end_ind, _num_frames = id_line.split(' ')
            return _vid_id, int(_start_ind), int(_end_ind), int(_num_frames)

        if vid_ids[0].split('/')[0] == 'train':
            vid_id_func = _train_get_vid_id
        elif vid_ids[0].split('/')[0] == 'val':
            vid_id_func = _val_get_vid_id
        else:
            raise ValueError("Unknown prefix in annoation txt file.")

        for id_line in vid_ids:
            # Probe first frame to get info
            vid_id, start_ind, end_ind, num_frames = vid_id_func(id_line)
            foldername = f'Data/VID/{vid_id}.JPEG'
            xml_path = Path(self.img_prefix
                            )/f'Annotations/VID/{vid_id}/{start_ind:06d}.xml'
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            vid_infos.append(
                dict(id=vid_id,
                     filename=foldername,
                     width=width,
                     height=height,
                     start_ind=start_ind,
                     end_ind=end_ind,
                     num_frames=num_frames))
        return vid_infos

    def get_ann_info(self, idx, frame_ids):
        vid_id = self.vid_infos[idx]['id']
        anns = []
        for frame_id in frame_ids:
            xml_path = Path(self.img_prefix
                            )/f'Annotations/VID/{vid_id}/{frame_id:06d}.xml'
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                label = self.cat2label[name]
                bnd_box = obj.find('bndbox')
                bbox = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ))
            else:
                bboxes = np.array(bboxes, ndmin=2) - 1
                labels = np.array(labels)
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0, ))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
                labels_ignore = np.array(labels_ignore)
            ann = dict(
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64),
                bboxes_ignore=bboxes_ignore.astype(np.float32),
                labels_ignore=labels_ignore.astype(np.int64))
            anns.append(ann)
        return anns

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_info = self.vid_infos[i]
            if vid_info['width'] / vid_info['height'] > 1:
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

    def get_frame_info(self, idx, frame_ids):
        vid_info = self.vid_infos[idx]
        vid_id = vid_info['id']
        width = vid_info['width']
        height = vid_info['height']
        img_infos = []
        for frame_id in frame_ids:
            img_id = f'{vid_id}/{frame_id:06d}'
            filename = f'Data/VID/{img_id}.JPEG'
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def select_clip(self, idx):
        vid_info = self.vid_infos[idx]
        start_ind = vid_info['start_ind']
        end_ind = vid_info['end_ind']
        num_frames = vid_info['num_frames']
        frame_ids = []

        if num_frames == self.seq_len:
            return list(range(start_ind, end_ind))
        elif num_frames < self.seq_len:
            # [1, 2, 3], seq_len = 7 -> [1, 1, 2, 2, 3, 3, 3]
            start_frame = start_ind
            repeat = self.seq_len // num_frames
            residue = self.seq_len % num_frames
            for frame_id in range(start_frame, num_frames):
                for _ in range(repeat):
                    frame_ids.append(frame_id)
            for _ in range(residue):
                frame_ids.append(num_frames - 1)
        else:
            if self.skip:
                skip = random.randint(1, int(num_frames / self.seq_len))
                start = random.randint(start_ind,
                                       num_frames - self.seq_len * skip)
                frame_ids = list(range(start, num_frames, skip))[:self.seq_len]
            else:
                start = np.random.randint(start_ind, end_ind - self.seq_len)
                frame_ids = list(range(start, start + self.seq_len))

        return frame_ids

    def select_test_clip(self, idx):
        vid_info = self.vid_infos[idx]
        start_ind = vid_info['start_ind']
        end_ind = vid_info['end_ind']
        end_ind = min(end_ind, start_ind + self.seq_len)
        return list(range(start_ind, end_ind))

    def prepare_train_img(self, idx):
        frame_ids = self.select_clip(idx)  # list of int
        img_infos = self.get_frame_info(idx, frame_ids)
        ann_infos = self.get_ann_info(idx, frame_ids)  # list of ann_info
        seq_results = []
        for img_info, ann_info in zip(img_infos, ann_infos):
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)
            results_dict = self.pipeline(results)
            seq_results.append(results_dict)
        seq_results_collated = seq_collate(seq_results)

        # Check at least one frame has annotation, since we did not use _filter_imgs()
        # during loading annotaion.
        sum_gts = sum([len(v) for v in seq_results_collated['gt_bboxes'].data])
        if sum_gts == 0:
            return None
        return seq_results_collated

    def prepare_test_img(self, idx):
        frame_ids = self.select_test_clip(idx)  # list of int
        img_infos = self.get_frame_info(idx, frame_ids)
        seq_results = []
        for img_info in img_infos:
            results = dict(img_info=img_info)
            self.pre_pipeline(results)
            results_dict = self.pipeline(results)
            seq_results.append(results_dict)
        seq_results_collated = seq_collate(seq_results)
        return seq_results_collated


def seq_collate(batch, samples_per_gpu=1):
    """ Modified from mmcv.collate
        Puts each data field into a tensor/DataContainer with outer dimension
        Seq Length.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """
    if not isinstance(batch, collections.Sequence):
        raise TypeError("{} is not supported.".format(batch.dtype))

    if isinstance(batch[0], DataContainer):
        assert len(batch) % samples_per_gpu == 0
        if batch[0].cpu_only:
            stacked = [sample.data for sample in batch]
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            assert isinstance(batch[0].data, torch.Tensor)

            if batch[0].pad_dims is not None:
                ndim = batch[0].dim()
                assert ndim > batch[0].pad_dims
                max_shape = [0 for _ in range(batch[0].pad_dims)]
                for dim in range(1, batch[0].pad_dims + 1):
                    max_shape[dim - 1] = batch[0].size(-dim)
                for sample in batch:
                    for dim in range(0, ndim - batch[0].pad_dims):
                        assert batch[0].size(dim) == sample.size(dim)
                    for dim in range(1, batch[0].pad_dims + 1):
                        max_shape[dim - 1] = max(max_shape[dim - 1],
                                                 sample.size(-dim))
                padded_samples = []
                for sample in batch:
                    pad = [0 for _ in range(batch[0].pad_dims * 2)]
                    for dim in range(1, batch[0].pad_dims + 1):
                        pad[2 * dim -
                            1] = max_shape[dim - 1] - sample.size(-dim)
                    padded_samples.append(
                        F.pad(
                            sample.data, pad, value=sample.padding_value))
                stacked = default_collate(padded_samples)
            elif batch[0].pad_dims is None:
                stacked = default_collate([
                    sample.data
                    for sample in batch
                ])
            else:
                raise ValueError(
                    'pad_dims should be either None or integers (1-3)')

        else:
            stacked = [sample.data for sample in batch]
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [seq_collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        collate_ret = {
            key: seq_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
        return collate_ret
    else:
        return default_collate(batch)
