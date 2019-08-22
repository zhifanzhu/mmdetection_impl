#
# ktw361@2019.5.26
#
from .registry import DATASETS


__author__ = 'ktw361'

from .coco import CocoDataset


@DATASETS.register_module
class VisDroneDataset(CocoDataset):

    CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
               'tricycle', 'awning-tricycle', 'bus', 'motor')
