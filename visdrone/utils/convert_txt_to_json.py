import argparse
import os
import os.path as osp
import hashlib
import tqdm
from functools import partial
from multiprocessing import Pool

import numpy as np
import mmcv
from visdrone.utils.get_image_size import get_image_size

""" This script convert visdron txt annotations to coco(mmdet) compatible format,
usage:
    python visdron/utils/convert_txt_to_json.py \
        --img-prefix=~/DATASETS/Drone2019/VisDrone2019-DET/VisDrone2018-DET-train \
        --out-file=~/DATASETS/Drone2019/VisDrone2019-DET/VisDrone2018-DET-train/annotations_train.json

Notes:
    field 'filename' include parent dir: 'images', so that (img_prefix + filename) can
    open image directly. i.e., 'filename'== images/<stem>.jpg

    field 'image_id' is create from stem, so that any who access image name cancompute:
        'image_id' = int(sha256(stem.encode('utf8')).hexdigest(), 16) % (10 ** 8)

    'id' field in 'annotations' is also obtained by:
        'id' = str(stem) + str(bbox)
        'id' = int(hashlib.sha256(id.encode('utf8')).hexdigest(), 16) % (10 ** 12)
"""


def create_categories():
    """ Create 'categories' field for coco up-most level"""

    CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
               'tricycle', 'awning-tricycle', 'bus', 'motor')
    categories = []
    for i, c in enumerate(CLASSES):
        categories.append(dict(
            supercategory=c,
            id=i + 1,
            name=c
        ))
    return categories


def _parse_txt_from_name(img_name, all_images, all_annos, img_dir):
    """
    :param img_dir: should be 'images'
    :return: (dict of img, list of dict of anno)
    """
    stem = img_name.split('.')[0]
    ann_name = stem + '.txt'
    img_full_path = osp.join(all_images, img_name)
    ann_full_path = osp.join(all_annos, ann_name)

    # get 'images' field for coco object
    file_name = img_dir + '/' + img_name
    image_id = int(hashlib.sha256(stem.encode('utf8')).hexdigest(), 16) % (10 ** 8)
    width, height = get_image_size(img_full_path)
    _images = dict(
        license=4,
        file_name=file_name,
        height=height,
        width=width,
        id=image_id,)

    # get 'annotations' field for coco object
    with open(ann_full_path, 'r') as f:
        lines = f.readlines()
        lines = [v.strip('\n') for v in lines]
        lines = [v.split(',') for v in lines]
        lines = np.asarray(lines)[:, :8].astype(np.int32)

    local_annotation = []
    for line in lines:
        x1, y1, w, h, sc, label, trun, occ = line
        label = int(label)
        if label == 0 or label == 11:
            # ignore ignore(0) and others(11)
            continue
        assert label > 0 and label < 11, 'Bad annotation label'
        w, h = int(w), int(h)
        # label is number, '0' is background
        bbox = np.asarray([x1, y1, w, h]).tolist()
        id = str(stem) + str(bbox)
        id = int(hashlib.sha256(id.encode('utf8')).hexdigest(), 16) % (10 ** 12)
        # add iscrowd
        iscrowd = 1 if int(occ) > 0 else 0

        _annotations = dict(
            image_id=image_id,
            bbox=bbox,
            category_id=label,
            id=id,
            area=w*h,
            iscrowd=iscrowd,
        )
        local_annotation.append(_annotations)
    return _images, local_annotation


def parse_txt(img_prefix, ann_dir='annotations', img_dir='images'):
    """ Create 'annotations' and 'images' field for up-most level

    Args:
        img_prefix: str, e.g.'~/DATASETS/Drone2019/VisDrone2019-DET/VisDrone2018-DET-train'
        ann_dir: dir name to annotations
        img_dir: dir name to images

    Returns:
        annotations:
            list of dict, for each bounding box
        images:
            list of dict, for each image
    """
    all_images = osp.join(img_prefix, img_dir)
    all_annos = osp.join(img_prefix, ann_dir)

    images = []
    annotations = []
    for img_name in tqdm.tqdm(os.listdir(all_images)):
        _images, _annotations = _parse_txt_from_name(img_name, all_images, all_annos,
                                                     img_dir)
        images.append(_images)
        annotations.extend(_annotations)
    return images, annotations


def mp_parse_txt(img_prefix, num_process, ann_dir='annotations', img_dir='images'):
    if num_process == 1:
        return parse_txt(img_prefix, ann_dir, img_dir)
    else:
        p = Pool(num_process)
        all_images = osp.join(img_prefix, img_dir)
        all_annos = osp.join(img_prefix, ann_dir)
        img_names = os.listdir(all_images)
        worker = partial(_parse_txt_from_name, all_images=all_images, all_annos=all_annos,
                         img_dir=img_dir)
        img_ann_list = list(tqdm.tqdm(p.imap(worker, img_names)))
        images, annotations = [], []
        for img, anno in img_ann_list:
            images.append(img)
            annotations.extend(anno)
        return images, annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone annotations to mmdetection format')
    parser.add_argument('--img-prefix', help='visdrone img root path')
    parser.add_argument('--out-file', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images, annotations = parse_txt(args.img_prefix)
    categories = create_categories()
    annotations = dict(
        images=images,
        annotations=annotations,
        categories=categories
    )
    mmcv.dump(annotations, args.out_file)


if __name__ == '__main__':
    main()
