import numpy as np
import os.path as osp

def single_det2txt(result, txt_file):
    """ write mmdetection result to txt_file

    Args:
        result: list(cls) of np array with [N, 5] of [x1, y1, x2, y2, score]
        txt_file: str specify output filename
    """
    with open(txt_file, 'w') as fid:
        for i, res in enumerate(result):
            cat_id = i + 1
            if len(res) == 0:
                continue
            for det in res:
                x1, y1, x2, y2, sc = det
                w = x2 - x1
                h = y2 - y1
                fid.writelines('%d,%d,%d,%d,%.4f,%d,-1,-1\n' % (
                    x1, y1, w, h, sc, cat_id
                ))


def single_txt2det(fid, num_classes=10):
    """ Returns detecton result from one txt(image)
    Args:
        fid: opened file handler.
    """
    lines = fid.readlines()
    lines = [v.strip('\n') for v in lines]
    lines = [v.split(',') for v in lines]
    dets = [[] for _ in range(num_classes)]
    for line in lines:
        x1, y1, w, h, sc, label, trun, occ = line
        label = int(label)
        if label == 0 or label == 11:
            continue
        assert label > 0 and label < 11, 'Bad label'
        x1, y1 = int(x1), int(y1)
        w, h = int(w), int(h)
        score = float(sc)
        x2 = x1 + w
        y2 = y1 + h
        bbox = np.asarray([x1, y1, x2, y2, score], dtype=np.float32)
        dets[label - 1].append(bbox)
    for i in range(len(dets)):
        if len(dets[i]) == 0:
            dets[i] = np.empty([0, 5])
        else:
            dets[i] = np.stack(dets[i], 0)
    return dets


def pkl2txt(dataset, results, save_dir):
    """
    :param dataset: Dataset object with img_infos, img_ids
    :param results: list(file idx) of list(class) of [N, 5]
    :return: dict from fstem to its merged_results,
        which is list(stem idx) of list(class) of [N, 5]
    """
    for info in dataset.img_infos:
        fname = info['file_name']
        f_id = info['id']
        txt_name = fname.replace('.jpg', '.txt').split('/')[-1]
        txt_name = osp.join(save_dir, txt_name)
        idx = dataset.img_ids.index(f_id)
        single_det2txt(results[idx], txt_name)
