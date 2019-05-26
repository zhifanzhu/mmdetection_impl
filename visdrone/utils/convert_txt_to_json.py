import argparse
import os
import os.path as osp
import hashlib

import numpy as np
import mmcv
from visdrone.utils.get_image_size import get_image_size


def parse_txt(stem, img_prefix, ann_dir='annotations', img_dir='images'):
    img_name = stem + '.jpg'
    ann_name = stem + '.txt'
    all_images = osp.join(img_prefix, img_dir)
    all_annos = osp.join(img_prefix, ann_dir)
    img_full_path = osp.join(all_images, img_name)
    ann_full_path = osp.join(all_annos, ann_name)

    filename = img_dir + '/' + img_name
    # image_id = np.int64(int(hashlib.sha1(stem.encode('utf8')).hexdigest(), 16) % (10 ** 8))
    image_id = int(hashlib.sha1(stem.encode('utf8')).hexdigest(), 16) % (10 ** 8)
    width, height = get_image_size(img_full_path)
    with open(ann_full_path, 'r') as f:
        lines = f.readlines()
        lines = [v.strip('\n') for v in lines]
        lines = [v.split(',') for v in lines]
        lines = np.asarray(lines)[:, :8].astype(np.int32)
    # bboxes = []
    # labels = []
    # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    annotation = []
    for line in lines:
        x1, y1, w, h, sc, label, trun, occ = line
        if label == 0 or label == 11:
            # ignore ignore(0) and others(11)
            continue
        assert label > 0 and label < 11, 'Bad annotation label'
        x2, y2 = x1 + w, y1 + h
        # label is number, '0' is background
        bbox = np.asarray([x1, y1, x2, y2]).tolist()
        id = str(stem) + str(bbox)
        id = int(hashlib.sha1(id.encode('utf8')).hexdigest(), 16) % (10 ** 12)

        # bboxes.append(bbox)
        # labels.append(label)
        anno = dict(
            image_id=image_id,
            bbox=bbox,
            category_id=label.tolist(),
            id=id,
        )
        annotation += [anno]

    # if not bboxes:
    #     bboxes = np.zeros((0, 4))
    #     labels = np.zeros((0, ))
    # else:
    #     bboxes = np.array(bboxes, ndmin=2) - 1
    #     labels = np.array(labels)

    # annotation = dict(
    #     filename=filename,
    #     width=width,
    #     height=height,
    #     ann=dict(
    #         bboxes=bboxes.astype(np.float32).tolist(),
    #         labels=labels.astype(np.int64).tolist(),
    #     )
    # )
    return annotation


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone annotations to mmdetection format')
    parser.add_argument('--img-prefix', help='visdrone img root path')
    parser.add_argument('--out-file', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    annotations = []
    for img in os.listdir(osp.join(args.img_prefix ,'images')):
        stem = img.split('.')[0]
        part_annotations = parse_txt(stem, args.img_prefix)
        # annotations += [part_annotations]
        annotations += part_annotations
    annotations = dict(
        annotations=annotations
    )
    mmcv.dump(annotations, args.out_file)


if __name__ == '__main__':
    main()
