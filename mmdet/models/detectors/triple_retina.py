from ..registry import DETECTORS
from .triple_single_stage import TripleSingleStageDetector


@DETECTORS.register_module
class TripleRetinaNet(TripleSingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 triple_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TripleRetinaNet, self).__init__(
            backbone, neck, bbox_head, triple_module,
            train_cfg, test_cfg, pretrained)
