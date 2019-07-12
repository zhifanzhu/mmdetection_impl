import os.path as osp
import mmcv
from visdrone.utils import MergeTxt


"""
    Like vidmerge.py, this script merge multiple final results given by
    different image detector.
"""


ORI_LIST = [
    # '/tmp/valpatch_libraLowThr6401500',

    # '/tmp/valpatch_A3640R1500E23',
    '/tmp/valpatch_ga6401500',
    '/tmp/valpatch_A3moreAnchors',
    '/tmp/valpatch_libraLTBS6401500',
    # '/tmp/valpatch_casv26401500',
    '/tmp/valpatch_casv2',

    # '/tmp/testpatch_A3moreAnchors',
    # '/tmp/testpatch_ga6401500',
    # '/tmp/testpatch_A36401500Real',
    # '/tmp/testpatch_libraLTBS6401500',
]

print(ORI_LIST)
nms_param = dict(iou_thr=0.15, max_det=500, score_thr=0.05)
print(nms_param)
ds = [MergeTxt.read_origin(v) for v in ORI_LIST]
d = MergeTxt.merge_dicts(ds, nms_param)
# out_dir = '/tmp/detout/'
out_dir = '/tmp/detout/'
mmcv.mkdir_or_exist(osp.expanduser(out_dir))
MergeTxt.dict2txt(d, out_dir=out_dir)
