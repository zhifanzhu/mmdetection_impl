from ..registry import DETECTORS
from .seq_single_stage import SeqSingleStageDetector


@DETECTORS.register_module
class SeqRetinaNet(SeqSingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 temporal_module,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SeqRetinaNet, self).__init__(
            backbone, neck, temporal_module, bbox_head,
            train_cfg, test_cfg, pretrained)
