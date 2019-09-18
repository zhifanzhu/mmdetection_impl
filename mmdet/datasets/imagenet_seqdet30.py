import inspect
import os.path as osp
import xml.etree.ElementTree as ET

import collections
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from mmdet.datasets.pipelines import RandomRatioCrop

from mmcv.parallel.data_container import DataContainer

import time
import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class SeqDET30Dataset(Dataset):

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116',
               'n02958343', 'n02402425', 'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
               'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049',)
    DATASET_NAME = 'vid'

    def __init__(self,
                 ann_file,
                 pipeline,
                 seq_len,
                 divisor,
                 min_size=None,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False):
        """

        Args:
            ann_file:
            pipeline:
            seq_len:
            divisor: int, since DET dataset are too large when images repeats to seq_len,
                use divisor to get len(ds) = len(original) // divisor.
            min_size:
            data_root:
            img_prefix:
            seg_prefix:
            proposal_file:
            test_mode:
        """
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.seq_len = seq_len
        self.divisor = divisor

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
        self.img_infos = self.load_annotations(self.ann_file)
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
        return len(self.img_infos) // self.divisor

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for id_line in img_ids:
            img_id, pos = id_line.split(' ')
            filename = 'Data/DET/{}.JPEG'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations/DET',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations/DET',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.cat2label:
                continue
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
        idx = min(idx * self.divisor, len(self) - 1)
        if self.test_mode:
            raise ValueError("Not allowed")
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def pipeline_with_state(self, data, state_list):
        if state_list is None:
            state_list = []
            for t in self.pipeline.transforms:
                if 'state' in inspect.getfullargspec(t).args:
                    data, state = t(data)
                    assert state is not None
                    state_list.append(state)
                else:
                    data = t(data)
                    state_list.append(None)
                if data is None:
                    return None, None
            return data, state_list
        else:
            for state, t in zip(state_list, self.pipeline.transforms):
                if state is None:
                    data = t(data)
                else:
                    data, _ = t(data, state)
                if data is None:
                    return None
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        seq_results = []
        for i in range(self.seq_len):
            results = dict(img_info=img_info, ann_info=ann_info)
            self.pre_pipeline(results)
            if i == 0:
                results_dict, trans_states = self.pipeline_with_state(
                    results, None)
            else:
                results_dict = self.pipeline_with_state(results, trans_states)
            if results_dict is None:
                return None
            seq_results.append(results_dict)
        seq_results_collated = seq_collate(seq_results)
        sum_gts = sum([len(v) for v in seq_results_collated['gt_bboxes'].data])
        if sum_gts == 0:
            return None
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
