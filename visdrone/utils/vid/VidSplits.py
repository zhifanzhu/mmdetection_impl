import argparse
import os
import os.path as osp
import numpy as np
from shutil import copyfile

import mmcv
from visdrone.utils import convert_txt_to_json
from visdrone.utils import ImgSplits

"""
TODO: gen json args

No need for multiprocess, since only reading txt.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone VID annotations to mmdetection format')
    parser.add_argument('--basepath', help='visdrone VID seq basepath')
    parser.add_argument('--split', help='Whether do split after convert to json')
    parser.add_argument('--with-label',help='whether have label, i.e, test set')
    args = parser.parse_args()
    return args


def seq2img(basepath, targetpath=None, with_label=True):
    """

    :param basepath: root of 'sequences' and annotations'
    :param targetpath: root of generated 'images' and 'annotations'
        if None, will add '-DET' to basepath by default
    :returns: fullpath of targetpath
    """
    ann_ext = '.txt'
    img_ext = '.jpg'

    seq_root = osp.join(basepath, 'sequences')
    ann_root = osp.join(basepath, 'annotations')
    all_seq = os.listdir(seq_root)

    if targetpath is None:
        targetpath = basepath + '-DET'
    target_img = osp.join(targetpath, 'images')
    target_ann = osp.join(targetpath, 'annotations')

    mmcv.mkdir_or_exist(target_img)
    if with_label:
        mmcv.mkdir_or_exist(target_ann)
    for seq_name in all_seq:
        if with_label:
            seq_ann = osp.join(ann_root, seq_name + ann_ext)
            labels = _get_seq_anno(seq_ann)
        frame_dir = osp.join(seq_root, seq_name)
        for frame in os.listdir(frame_dir):
            stem = frame.replace(img_ext, '')
            frame_ind = int(stem)
            dst_stem = '_'.join([seq_name, stem])
            src = osp.join(frame_dir, frame)
            dst = dst_stem + img_ext
            dst = osp.join(target_img, dst)
            copyfile(src, dst)

            # write label
            if with_label:
                frame_ann = labels[labels[:, 0] == frame_ind]
                ann_dst = dst_stem + ann_ext
                ann_dst = osp.join(target_ann, ann_dst)
                with open(ann_dst, 'w') as fid:
                    for gt in frame_ann:
                        _, trackid, x1, y1, w, h, sc, label, trun, occ = gt
                        fid.writelines('%d,%d,%d,%d,%d,%d,%d,%d\n' % (
                            x1, y1, w, h, sc, label, trun, occ
                        ))
    return targetpath


def _get_seq_anno(ann_file):
    with open(ann_file, 'r') as f:
        labels = f.readlines()
        labels = [v.strip('\n') for v in labels]
        labels = [v.split(',') for v in labels]
        labels = np.asarray(labels)[:, :10].astype(np.int32)
    return labels


def main():
    args = parse_args()
    if args.with_label == 'True':
        with_label = True
    elif args.with_label == 'False':
        with_label = False
    else:
        raise ValueError("with_label either True or False")
    target_path = seq2img(args.basepath, with_label=with_label)
    if with_label:
        print('Converting to json...')
        images, annotations = convert_txt_to_json.mp_parse_txt(target_path,
                                                               num_process=8)
        categories = convert_txt_to_json.create_categories()
        annotations = dict(
            images=images,
            annotations=annotations,
            categories=categories
        )
        out_file = osp.join(target_path, 'annotations_val.json')
        mmcv.dump(annotations, out_file)
        print('Done')

    if args.split:
        patch_path = target_path + '-' + args.split
        subsizes = (int(args.split), )
        num_process = 8
        ImgSplits.split_multi_sizes(basepath=target_path,
                                    outpath=patch_path,
                                    num_process=num_process,
                                    subsizes=subsizes,
                                    gap=128,
                                    with_label=with_label)
        if with_label:
            print('Converting to json...')
            images, annotations = convert_txt_to_json.mp_parse_txt(patch_path,
                                                                   num_process)
            categories = convert_txt_to_json.create_categories()
            annotations = dict(
                images=images,
                annotations=annotations,
                categories=categories
            )
            out_file = 'annotations_val.json'
            out_file = osp.join(patch_path, out_file)
            mmcv.dump(annotations, out_file)
            print('Done')


if __name__ == '__main__':
    main()
