#
# ktw361@2019.5.26
#

__author__ = 'ktw361'

from .coco import CocoDataset


class VisDroneDataset(CocoDataset):

    CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
               'tricycle', 'awning-tricycle', 'bus', 'motor')
