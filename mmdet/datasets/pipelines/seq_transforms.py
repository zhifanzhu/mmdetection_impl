import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES


@PIPELINES.register_module
class SeqRandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def __call__(self, results, state=None):
        if 'flip' not in results:
            if state is None:
                flip = True if np.random.rand() < self.flip_ratio else False
                state = dict(flip=flip)
            else:
                flip = state['flip']
            results['flip'] = flip

        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
        return results, state

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class SeqRandomCrop(object):
    """Random crop the image & bboxes.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results, state=None):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        if state is None:
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            state = dict(offset_h=offset_h, offset_w=offset_w)
        else:
            offset_h = state['offset_h']
            offset_w = state['offset_w']
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i in valid_inds:
                    gt_mask = results['gt_masks'][i][crop_y1:crop_y2, crop_x1:
                                                     crop_x2]
                    valid_gt_masks.append(gt_mask)
                results['gt_masks'] = valid_gt_masks

        return results, state

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class SeqPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.contrast_range = contrast_range
        self.saturation_range =saturation_range

    def __call__(self, results, state=None):
        if state is None:
            delta_randint = random.randint(2)
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            mode = random.randint(2)
            mode_randint = random.randint(2)
            alpha = random.uniform(self.contrast_lower,
                                   self.contrast_upper)
            sat_randint = random.randint(2)
            sat_multiplier = random.uniform(self.saturation_lower,
                                            self.saturation_upper)
            hue_randint = random.randint(2)
            hue_multiplier = random.uniform(-self.hue_delta, self.hue_delta)
            contrast_randint = random.randint(2)
            contrast_alpha = random.uniform(self.contrast_lower,
                                            self.contrast_upper)
            perm_randint = random.randint(2)
            perm = random.permutation(3)

            state = dict(
                delta_randint=delta_randint,
                delta=delta,
                mode=mode,
                mode_randint=mode_randint,
                alpha=alpha,
                sat_randint=sat_randint,
                sat_multiplier=sat_multiplier,
                hue_randint=hue_randint,
                hue_multiplier=hue_multiplier,
                contrast_randint=contrast_randint,
                contrast_alpha=contrast_alpha,
                perm_randint=perm_randint,
                perm=perm)
        else:
            delta_randint = state['delta_randint']
            delta = state['delta']
            mode = state['mode']
            mode_randint = state['mode_randint']
            alpha = state['alpha']
            sat_randint = state['sat_randint']
            sat_multiplier = state['sat_multiplier']
            hue_randint = state['hue_randint']
            hue_multiplier = state['hue_multiplier']
            contrast_randint = state['contrast_randint']
            contrast_alpha = state['contrast_alpha']
            perm_randint = state['perm_randint']
            perm = state['perm']

        img = results['img']
        # random brightness
        if delta_randint:
            delta = delta
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = mode
        if mode == 1:
            if mode_randint:
                alpha = alpha
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if sat_randint:
            img[..., 1] *= sat_multiplier

        # random hue
        if hue_randint:
            img[..., 0] += hue_multiplier
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if contrast_randint:
                alpha = contrast_alpha
                img *= alpha

        # randomly swap channels
        if perm_randint:
            img = img[..., perm]

        results['img'] = img
        return results, state

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class SeqExpand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
    """

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range

    def __call__(self, results, state=None):
        if state is None:
            ret_randint = random.randint(2)
        else:
            ret_randint = state['ret_randint']

        if ret_randint:
            state = dict(ret_randint=ret_randint)
            return results, state

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        if state is None:
            ratio = random.uniform(self.min_ratio, self.max_ratio)
        else:
            ratio = state['ratio']
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        if state is None:
            left = int(random.uniform(0, w * ratio - w))
            top = int(random.uniform(0, h * ratio - h))
        else:
            left = state['left']
            top = state['top']
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes
        state = dict(
            ret_randint=ret_randint,
            ratio=ratio,
            left=left,
            top=top)
        return results, state

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={})'.format(
            self.mean, self.to_rgb, self.ratio_range)
        return repr_str


@PIPELINES.register_module
class SeqMinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_ious = min_ious
        self.min_crop_size = min_crop_size

    def __call__(self, results, state=None):
        if state is not None:
            return self._call_with_state(results, state)

        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            # Force return origin for no annotation.
            if len(boxes) == 0:
                mode = 1
            state = dict(mode=mode)
            if mode == 1:
                return results, state

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels

                state['new_w'] = new_w
                state['new_h'] = new_h
                state['left'] = left
                state['top'] = top
                return results, state

    @staticmethod
    def _call_with_state(results, state):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        mode = state['mode']
        if mode == 1:
            return results, state

        new_w = state['new_w']
        new_h = state['new_h']
        left = state['left']
        top = state['top']

        patch = np.array(
            (int(left), int(top), int(left + new_w), int(top + new_h)))

        # center of boxes should inside the crop img
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] > patch[0]) * (
                center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                       center[:, 1] < patch[3])
        boxes = boxes[mask]
        labels = labels[mask]

        # adjust boxes
        img = img[patch[1]:patch[3], patch[0]:patch[2]]
        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
        boxes -= np.tile(patch[:2], 2)

        results['img'] = img
        results['gt_bboxes'] = boxes
        results['gt_labels'] = labels
        return results, state

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str
