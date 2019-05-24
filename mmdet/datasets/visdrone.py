#
# ktw361@2019.5.22
#

__author__ = 'ktw361'

import os
import os.path as osp

import numpy as np

from .custom import CustomDataset

from visdrone.utils.get_image_size import get_image_size
""" use third party get_image_size() function rather than PIL.Image.open, 3x faster:
    Measured on 500 images by %timeit:
    this: 7.92 ms ± 71.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    PIL: 23.6 ms ± 87.1 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
"""


class VisDroneDataset(CustomDataset):

    CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
               'tricycle', 'awning-tricycle', 'bus', 'motor')

    def __init__(self, **kwargs):
        super(VisDroneDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file=None):
        assert ann_file is None, 'ann_file should be None: we read from diretory.'
        img_infos = []
        all_images = osp.join(self.img_prefix, 'images')
        for img_name in os.listdir(all_images):
            img_id = img_name.split('.')[0]
            full_path = osp.join(all_images, img_name)
            filename = 'images/{}'.format(img_name)
            width, height = get_image_size(full_path)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        txt_path = osp.join(self.img_prefix, 'annotations',
                            '{}.txt'.format(img_id))
        # Read annotation
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            lines = [v.strip('\n') for v in lines]
            lines = [v.split(',') for v in lines]
            lines = np.asarray(lines)[:, :8].astype(np.int32)

        bboxes = []
        labels = []
        bboxes_trunc = []
        labels_trunc = []
        bboxes_occlu = []
        labels_occlu = []

        # <object_category> here means number, but mmdetection
        # convention's cat is string of name.
        # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        for line in lines:
            x1, y1, w, h, sc, label, trun, occ = line
            x2, y2 = x1 + w, y1 + h
            # label is number, '0' is background
            bbox = [x1, y1, x2, y2]

            bboxes.append(bbox)
            labels.append(label)
            if trun == 1:
                bboxes_trunc.append(bbox)
                labels_trunc.append(label)
            if occ == 1:
                bboxes_occlu.append(bbox)
                labels_occlu.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        if not bboxes_trunc:
            bboxes_trunc = np.zeros((0, 4))
            labels_trunc = np.zeros((0, ))
        else:
            bboxes_trunc = np.array(bboxes_trunc, ndmin=2) - 1
            labels_trunc = np.array(labels_trunc)

        if not bboxes_occlu:
            bboxes_occlu = np.zeros((0, 4))
            labels_occlu = np.zeros((0, ))
        else:
            bboxes_occlu = np.array(bboxes_occlu, ndmin=2) - 1
            labels_occlu = np.array(labels_occlu)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_trunc=bboxes_trunc.astype(np.float32),
            labels_trunc=labels_trunc.astype(np.int64),
            bboxes_occlu=bboxes_occlu.astype(np.float32),
            labels_occlu=bboxes_occlu.astype(np.int64),
        )
        return ann
