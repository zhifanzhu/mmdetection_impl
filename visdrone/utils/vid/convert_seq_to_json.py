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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone VID annotations to mmdetection format')
    parser.add_argument('--root', help='visdrone VID seq root path')
    parser.add_argument('--mode', help='train, val or test')
    parser.add_argument('--num-process', help='num process')
    args = parser.parse_args()
    return args


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


def _parse_single_seq_from(frame_name, frame_content, seq_name, mode_root, seq_dir='sequences'):
    """
        path: mode_root/seq_dir/seq_name/frame_name.
    :param frame_name: '00000081.jpg' like
    :param frame_content: [N, 8] array, where we need to find by
        frame_content[:, 0] == int(img_stem_name),
    :param seq_name: 'uav00000086_0000_v' like
    :param seq_dir: should be 'sequences'
    :param mode_root: expanded full path to seq root ('xxx/VisDrone-VID-train/' like)
    :return: (dict of img, list of dict of anno)
    """
    frame_idx = int(frame_name.split('.')[0])
    frame_full_path = osp.join(mode_root, seq_dir, seq_name, frame_name)
    frame_content = frame_content[frame_content[:, 0] == frame_idx]

    # get 'images' field for coco object
    file_name = '/'.join([seq_dir, seq_name, frame_name])
    stem = '/'.join([seq_name, str(frame_idx)])  # for hash only
    image_id = int(hashlib.sha256(stem.encode('utf8')).hexdigest(), 16) % (10 ** 8)
    width, height = get_image_size(frame_full_path)
    _images = dict(
        license=4,
        file_name=file_name,
        height=height,
        width=width,
        id=image_id,)

    local_annotation = []
    for line in frame_content:
        frame_id, track_id, x1, y1, w, h, sc, label, trun, occ = line
        label = int(label)
        if label == 0 or label == 11:
            # ignore ignore(0) and others(11)
            continue
        assert 0 < label < 11, 'Bad annotation label'
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


def parse_seq(mode_root, ann_dir='annotations', seq_dir='sequences'):
    """ Create 'annotations' and 'images' field for up-most level

    Args:
        mode_root: str, e.g.'~/DATASETS/Drone2019/VisDrone2019-VID/VisDrone2018-VID-train'
        ann_dir: dir name to annotations
        seq_dir: dir name to sequences

    Returns:
        annotations:
            list of dict, for each bounding box
        frames:
            list of dict, for each frame across diff sequences
    """
    all_seqs = osp.join(mode_root, seq_dir)
    all_annos = osp.join(mode_root, ann_dir)

    frames = []  # images
    annotations = []
    print('total {} sequences'.format(len(os.listdir(all_seqs))))
    for i, seq_name in enumerate(os.listdir(all_seqs)):
        print('processing {}th seq'.format(i + 1))
        all_frames = osp.join(all_seqs, seq_name)

        # read frame content and dispatch
        ann_file = osp.join(all_annos, seq_name + '.txt')
        with open(ann_file, 'r') as f:
            labels = f.readlines()
            labels = [v.strip('\n') for v in labels]
            labels = [v.split(',') for v in labels]
            labels = np.asarray(labels)[:, :10].astype(np.int32)

        for fidx, frame_name in enumerate(tqdm.tqdm(os.listdir(all_frames))):
            # frame_content = labels[labels[:, 0] == fidx]
            _frame, _annotations = _parse_single_seq_from(frame_name, labels, seq_name, mode_root,
                                                          seq_dir=seq_dir)
            frames.append(_frame)
            annotations.extend(_annotations)
    return frames, annotations


def mp_parse_seq(mode_root, num_process, ann_dir='annotations', seq_dir='sequences'):
    if num_process == 1:
        return parse_seq(mode_root, ann_dir, seq_dir)
    else:
        p = Pool(num_process)
        all_seqs = osp.join(mode_root, seq_dir)
        all_annos = osp.join(mode_root, ann_dir)
        seq_names = os.listdir(all_seqs)

        frames = []  # images
        annotations = []
        print('total {} sequences'.format(len(os.listdir(all_seqs))))
        for i, seq_name in enumerate(os.listdir(all_seqs)):
            print('processing {}th seq'.format(i + 1))
            all_frames = osp.join(all_seqs, seq_name)

            # read frame content and dispatch
            ann_file = osp.join(all_annos, seq_name + '.txt')
            with open(ann_file, 'r') as f:
                labels = f.readlines()
                labels = [v.strip('\n') for v in labels]
                labels = [v.split(',') for v in labels]
                labels = np.asarray(labels)[:, :10].astype(np.int32)

            worker = partial(_parse_single_seq_from, frame_content=labels,
                             seq_name=seq_name, mode_root=mode_root, seq_dir=seq_dir)
            frame_names = os.listdir(all_frames)
            frame_ann_list = list(tqdm.tqdm(p.imap(worker, frame_names)))
            for frm, anno in frame_ann_list:
                frames.append(frm)
                annotations.extend(anno)

        return frames, annotations


def convert_to_json(root, mode, num_process, json_prefix='annotations_'):

    out_json = json_prefix + mode + '.json'

    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        mode = 'VisDrone2018-VID-train'
    elif mode == 'val':
        mode = 'VisDrone2018-VID-val'
    elif mode == 'test':
        mode = 'VisDrone2018-VID-test-challenge'
    else:
        raise KeyError('mode incorrect')

    mode_root = osp.expanduser(osp.join(root, mode))
    print('Converting to json...')
    frames, annotations = mp_parse_seq(mode_root, num_process)
    categories = create_categories()
    annotations = dict(
        images=frames,
        annotations=annotations,
        categories=categories
    )
    out_json = osp.join(mode_root, out_json)
    mmcv.dump(annotations, out_json)
    print('Done')


if __name__ == '__main__':
    args = parse_args()
    convert_to_json(args.root, args.mode, int(args.num_process))
