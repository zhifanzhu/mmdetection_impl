from abc import abstractmethod

from mmdet.core import auto_fp16
from mmdet.models.detectors import BaseDetector

"""
See note in track_single_stage.py
"""


class TrackBaseDetector(BaseDetector):
    def __init__(self):
        super(TrackBaseDetector, self).__init__()

    @auto_fp16(apply_to=('img', 'ref_img'))
    @abstractmethod
    def forward_train(self, img, ref_img, img_metas, ref_img_metas, **kwargs):
        """
        Args:
            imgs (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            ref_imgs (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            ref_img_metas: same as img_metas

             **kwargs: specific to concrete implementation
        """
        pass

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            ref_img = kwargs.pop('ref_img')
            ref_img_meta = kwargs.pop('ref_img_meta')
            return self.forward_train(img, ref_img, img_meta, ref_img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)
