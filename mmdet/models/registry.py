from mmdet.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
TEMPORAL_MODULE = Registry('temporal_module')
PAIR_MODULE = Registry('pair_module')
TRIPLE_MODULE = Registry('triple_module')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
