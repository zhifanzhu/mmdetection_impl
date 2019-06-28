import argparse
import os
import os.path as osp
import tqdm
from multiprocessing import Pool
from functools import partial

import numpy as np
import mmcv
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from visdrone.utils import convert_txt_to_json

"""
    Files(.jpg, and .txt) will be created to VisDrone2018-DET-{mode}_patch' by default.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split VisDrone annotations and convert to mmdetection format')
    parser.add_argument('--root', help='visdrone img root path')
    parser.add_argument('--size', type=int, default=None, help='if specified, only generate this size')
    parser.add_argument('--mode', help='train, val or test')
    parser.add_argument('--num-process', help='num process')
    args = parser.parse_args()
    return args


def split_single_wrap(img_path, split_base):
    split_base.split_and_save_by_imgpath(img_path=img_path)


class Splitbase(object):

    def __init__(self,
                 basepath,
                 outpath,
                 gap=128,
                 subsize=1024,
                 thresh=0.7,
                 ext='.jpg',
                 label_ext='.txt',
                 num_process=1):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'annotations'
        :param gap: overlap between two patches
        :param subsize: int or tuple, subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of
            split
        :param ext: ext for the image format
        :param label_ext: ext for label format
        """
        assert ext in ['.jpg', '.png']
        assert label_ext in ['.txt']

        self.basepath = osp.expanduser(basepath)
        self.outpath = osp.expanduser(outpath)
        self.gap = gap
        self.subsize = subsize
        self.slide = subsize - gap
        self.thresh = thresh
        self.imagepath = osp.join(self.basepath, 'images')
        self.labelpath = osp.join(self.basepath, 'annotations')
        self.outimagepath = osp.join(self.outpath, 'images')
        self.outlabelpath = osp.join(self.outpath, 'annotations')
        self.ext = ext
        self.label_ext = label_ext
        self.num_process = num_process
        self.pool = Pool(num_process)

        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outlabelpath):
            os.makedirs(self.outlabelpath)

    def __getstate__(self):
        """
            Work around 'pool objects cannot be passed between processes or pickled'
        """
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def save_one_patch(self, subimg, sublabels, subname):
        img_save_name = subname + self.ext
        img_out = osp.join(self.outimagepath, img_save_name)
        mmcv.imwrite(subimg, img_out)
        # save label
        label_save_name = subname + self.label_ext
        label_out = osp.join(self.outlabelpath, label_save_name)
        with open(label_out, 'w') as fid:
            for i, label in enumerate(sublabels):
                x, y, w, h, sc, cat, truc, occ = label
                fid.writelines('%d,%d,%d,%d,%d,%d,%d,%d\n' % (
                    x, y, w, h, sc, cat, truc, occ))

    def _getsplitcoors(self, img_shape):
        """
        :return: [k, 4] np array in [x1, y1, x2, y2]
        """
        coors = []
        height, width = img_shape
        left, up = 0, 0
        while left < width:
            if left + self.subsize >= width:
                left = max(width - self.subsize, 0)
            up = 0
            while up < height:
                if up + self.subsize >= height:
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, width - 1)
                down = min(up + self.subsize, height - 1)
                coors.append((left, up, right, down))
                if up + self.subsize >= height:
                    break
                else:
                    up = up + self.slide
            if left + self.subsize >= width:
                break
            else:
                left = left + self.slide
        return np.asarray(coors)

    def split_img_labels(self, img, labels):
        """
            split a single image and ground truth.
            Crop & prune boxes happens in this stage.
        :param img:  np array
        :param labels: [N, 8] np array, [x, y, w, h, sc, cat, trunc occ]
        :return: list of (img, labels, coors) tuple for each patch
            note: [xywh] in , [xywh] out
        """
        img_h, img_w, _ = img.shape
        coors = self._getsplitcoors(img_shape=(img_h, img_w))
        patch_list = mmcv.imcrop(img, coors)
        labels_list = []
        invalid_inds = []

        for i, coor in enumerate(coors):
            labels_i = np.copy(labels)
            boxes = labels_i[:, :4].astype(np.float32)
            boxes = xywh2xyxy(boxes)
            overlaps = bbox_overlaps(
                boxes.reshape(-1, 4),
                coor.reshape(-1, 4),
                mode='iof').reshape(-1)  # note: 'iof' = ov/area_1.

            # Prune boxes outside image
            inside_inds = (overlaps > self.thresh).nonzero()
            if len(inside_inds) == 0:
                # print('  inside_inds == 0')
                labels_list.append(np.empty((0, 8), np.int32))
                invalid_inds.append(i)
                continue

            boxes = boxes[inside_inds]
            labels_i = labels_i[inside_inds]

            # center of boxes should inside the crop img
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            center_inds = (center[:, 0] > coor[0]) * (
                    center[:, 1] > coor[1]) * (center[:, 0] < coor[2] - 1) * (
                           center[:, 1] < coor[3] - 1)
            if not center_inds.any():
                # print('  center_inds == 0')
                labels_list.append(np.empty((0, 8), np.int32))
                invalid_inds.append(i)
                continue

            boxes = boxes[center_inds]
            labels_i = labels_i[center_inds]

            # adjust boxes
            boxes[:, 2:] = boxes[:, 2:].clip(max=coor[2:])
            boxes[:, :2] = boxes[:, :2].clip(min=coor[:2])
            boxes -= np.tile(coor[:2], 2)
            boxes = xyxy2xywh(boxes)
            labels_i[:, :4] = boxes.astype(np.int32)
            labels_list.append(labels_i)

        assert len(labels_list) == len(coors)
        assert len(labels_list) == len(patch_list)
        valid_inds = list(set(range(len(coors))) - set(invalid_inds))
        patch_list = [patch_list[i] for i in valid_inds]
        labels_list = [labels_list[i] for i in valid_inds]
        coors = [coors[i] for i in valid_inds]

        ret = []
        for subimg, sublabels, coor in zip(patch_list,
                                           labels_list,
                                           coors):
            ret.append((subimg, sublabels, coor))
        return ret

    def split_and_save_by_imgpath(self, img_path):
        """
            This function will do:
            1. read img and labels according to img_path, into np array
            2. call split_img_labels to get (img, label, coor)
            3. save patches
        """
        name = img_path.split('.')[0]
        label_path = osp.join(self.labelpath, img_path.replace(self.ext, self.label_ext))
        img_path = osp.join(self.imagepath, img_path)
        img = mmcv.imread(img_path)
        img_h, img_w, _ = img.shape
        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = [v.strip('\n') for v in labels]
            labels = [v.split(',') for v in labels]
            labels = np.asarray(labels)[:, :8].astype(np.int32)
        subimg_labels = self.split_img_labels(img, labels)

        # step 3, save patches
        for subimg, sublabels, coor in subimg_labels:
            coor = xyxy2xywh(coor[None, :]).squeeze(0)
            subname = '_'.join([str(v) for v in coor])  # x_y_w_h
            subname = '_'.join([str(img_h), str(img_w), subname])  # join image ori shape
            subname = '__'.join([name, subname])  # concat fid with double underscore
            self.save_one_patch(subimg, sublabels, subname)

    def split_and_save(self):
        """
            This function is a wrapper of split_and_save_by_imgpath
        """
        img_paths = os.listdir(self.imagepath)
        prog_bar = mmcv.ProgressBar(len(img_paths))
        for img_path in img_paths:
            self.split_and_save_by_imgpath(img_path)
            prog_bar.update()

    def mp_split_and_save(self):
        """
            multiple process split_and_save()
        """
        if self.num_process == 1:
            self.split_and_save()
        else:
            img_paths = os.listdir(self.imagepath)
            prog_bar = mmcv.ProgressBar(len(img_paths))
            def update():
                prog_bar.update()

            worker = partial(split_single_wrap, split_base=self)
            _ = list(tqdm.tqdm(self.pool.imap(worker, img_paths), total=len(img_paths)))


def xywh2xyxy(boxes):
    """
    :param boxes: [N, 4] np array, [x1, y1, w, h]
    :return: [N, 4] with [x1, y1, x2, y2]
    """
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def xyxy2xywh(boxes):
    boxes[:, 2:] -= boxes[:, :2]
    return boxes


def split_multi_sizes(basepath,
                      outpath,
                      num_process,
                      subsizes=(512, 640, 768, 896, 1024),
                      gap=128):
    for subsz in subsizes:
        print('Split size: {}'.format(subsz))
        S = Splitbase(basepath, outpath, subsize=subsz, gap=gap, num_process=num_process)
        S.mp_split_and_save()


def split_mode_then_convert_to_json(root,
                                    mode,
                                    size=None,
                                    num_process=3,
                                    json_prefix='annotations_'):
    assert mode in ['train', 'val', 'test']
    out_file = json_prefix + mode + '.json'
    if mode == 'train':
        mode = 'VisDrone2018-DET-train'
    elif mode == 'val':
        mode = 'VisDrone2018-DET-val'
    elif mode == 'test':
        mode = 'VisDrone2018-DET-test-challenge'
    else:
        raise KeyError('mode incorrect')

    basepath = osp.expanduser(osp.join(root, mode))
    # Do split
    if size is None:
        subsizes = (512, 640, 768, 896, 1024)
        save_ext = '-patch'
    else:
        subsizes=(size, )
        save_ext = '-' + str(size)
    patchpath = basepath + save_ext
    split_multi_sizes(basepath, patchpath, num_process, subsizes)

    # Now that patchpath is not empty, we can do conversion.
    print('Converting to json...')
    images, annotations = convert_txt_to_json.mp_parse_txt(patchpath, num_process)
    categories = convert_txt_to_json.create_categories()
    annotations = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )
    out_file = osp.join(patchpath, out_file)
    mmcv.dump(annotations, out_file)
    print('Done')


if __name__ == '__main__':
    args = parse_args()
    split_mode_then_convert_to_json(args.root, args.mode,
                                    size=args.size,
                                    num_process=int(args.num_process))
