from ..registry import DETECTORS
from .pair_x1_single_stage import PairX1SingleStageDetector


@DETECTORS.register_module
class PairX1RetinaNet(PairX1SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PairX1RetinaNet, self).__init__(
            backbone, neck, bbox_head,
            train_cfg, test_cfg, pretrained)
