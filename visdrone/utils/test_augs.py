#
# ktw361@2019.6.9
#

__author__ = 'ktw361'

import numpy as np

import mmcv


def get9crops(img, ori_shape=None):
    """
    9 crops, No flip. Crop size = (h/2, w/2)

    Args:
        img: np array [H, W, 3]
        ori_shape: (H, W)
    Returns:
        patches: list of 9 cropped patches
        coors: np array [9, 4] of coordinates used for cropping. (x1, y1, x2, y2)
    """
    if ori_shape is None:
        H, W, _ = img.shape
    else:
        H, W = ori_shape
    crop_wh = (W / 2, H / 2)  # note the order: W then H
    x1 = np.array([0, W / 4, W / 2])
    y1 = np.array([0, H / 4, H / 2])
    topleft = np.stack(np.meshgrid(x1, y1), -1).reshape(-1, 2)
    bottomright = topleft + crop_wh
    coors = np.concatenate([topleft, bottomright], -1)
    return mmcv.imcrop(img, coors), coors


def get10crops(img, ori_shape=None):
    """
    10 crop = 9 crop + 1 original
    Args:
        img: np array [H, W, 3]
        ori_shape: (H, W)
    Returns:
        patches: list of 10 patches
            The last one is orignial image.
        coors: [10, 4] array, (x1, y1, x2, y2)
    """
    if ori_shape is None:
        H, W, _ = img.shape
    else:
        H, W = ori_shape
    patches, coors = get9crops(img, ori_shape)
    patches.append(img)
    coors = np.concatenate([
        coors,
        [[0, 0, W, H]],
    ])
    return patches, coors


def bbox_revert_crop(bboxes, coor):
    """
    Args:
        bboxes: [N, 4] det of one crop.
        coor: [4, ] coor of that crop.
    Returns:
        bboxes: [N, 4] reverted to original image.
    """
    bboxes += np.tile(coor[:2], 2)
    return bboxes


def bbox_score_revert_crop(preds, coor):
    """
    Args:
        preds: [N, 5] det of one crop. plus a column of score.
        coor: [4, ] coor of that crop.
    Returns:
        bboxes: [N, 4] reverted to original image.
    """
    bboxes = preds[:, :4]
    preds[:, :4] = bbox_revert_crop(bboxes, coor)
    return preds
