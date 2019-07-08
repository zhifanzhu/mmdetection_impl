import os.path as osp
import mmcv
from visdrone.utils import result_utils
from visdrone.utils import MergeTxt

SEQ_LIST = [
    '/tmp/testpatch_ga6401500',
    '/tmp/testpatch_A36401500Real',
    '/tmp/testpatch_libraLowThr6401500',
]

print(SEQ_LIST)
nms_param=dict(iou_thr=0.33, max_det=350, score_thr=0.05)
print(nms_param)


# 1 split seq into images
seq_ds = [result_utils.single_seq2res(v) for v in SEQ_LIST]
# 2 merge images
d = MergeTxt.merge_dicts(seq_ds, nms_param)

# 3 aggregate
out_dir = '/tmp/vidout/'
mmcv.mkdir_or_exist(osp.expanduser(out_dir))


def dict2seq(d, save_dir):
    tmp_dict = dict()
    for img_name, img_dets in d.items():
        info = img_name.split('_')
        seq_name = info[:-1]
        frame_ind = int(info[-1])
        if seq_name not in tmp_dict:
            tmp_dict[seq_name] = {frame_ind: img_dets}
        else:
            seq_name[seq_name][frame_ind] = img_dets
    MergeTxt.save_seq_results(save_dir, tmp_dict)


dict2seq(d, out_dir)
