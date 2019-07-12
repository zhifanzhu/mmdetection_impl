import os.path as osp
import mmcv
from visdrone.utils import result_utils
from visdrone.utils import MergeTxt

"""
    Script that merge multiple final results given by different detectors.
"""


SEQ_LIST = [
    # '~/val896_A2enhance',
    # '/tmp/valori_gavidseq/',
    # '~/val896_A2patch1500E1/',
    # '/tmp/valori_A2Ehvid133E10seq/',
    '~/test896_A2enhance',
    '/tmp/testori_gavid1500seq/',
    '~/test896_A2patch1500E1/',
    '/tmp/testori_A2Ehvid133E10seq/',
]

print(SEQ_LIST)
nms_param = dict(iou_thr=0.5, max_det=100, score_thr=0.05)
print(nms_param)


# 1 split seq into images
SEQ_LIST = [osp.expanduser(v) for v in SEQ_LIST]
seq_ds = [result_utils.single_seq2res(v) for v in SEQ_LIST]
# 2 merge images
d = MergeTxt.merge_dicts(seq_ds, nms_param)

# 3 aggregate
out_dir = '/tmp/vidS123G1/'
print(out_dir)
mmcv.mkdir_or_exist(osp.expanduser(out_dir))


def dict2seq(d, save_dir):
    tmp_dict = dict()
    for img_name, img_dets in d.items():
        info = img_name.split('_')
        seq_name = '_'.join(info[:-1])
        frame_ind = int(info[-1])
        if seq_name not in tmp_dict:
            tmp_dict[seq_name] = {frame_ind: img_dets}
        else:
            tmp_dict[seq_name][frame_ind] = img_dets
    MergeTxt.save_seq_results(tmp_dict, save_dir)


dict2seq(d, out_dir)
