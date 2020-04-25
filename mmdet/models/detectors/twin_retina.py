from ..registry import DETECTORS
from .twin_single_stage import TwinSingleStageDetector


@DETECTORS.register_module
class TwinRetinaNet(TwinSingleStageDetector):

    def __init__(self,
                 backbone,
                 twin,
                 twin_load_from,
                 neck,
                 bbox_head,
                 pair_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwinRetinaNet, self).__init__(
            backbone, twin, twin_load_from, neck, bbox_head, pair_module,
            train_cfg, test_cfg, pretrained)
