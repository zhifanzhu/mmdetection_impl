#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

d=work_dirs/cascade/x101_640_v2
CONFIG=$d/cas_x101_640v2.py
CHECKPOINT=$d/latest.pth
GPUS=8
I="~/DATASETS/Drone2019/VisDrone2019-DET/VisDrone2018-DET-test-challenge-patch/images"
# I="~/DATASETS/Drone2019/VisDrone2019-DET/VisDrone2018-DET-val-patch/images"
# I="~/DATASETS/Drone2019/VisDrone2019-DET/VisDrone2018-DET-val/images"
O="/tmp/testpatch_casv2"
# O="/tmp/valpatch_casv2"

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    tmptest.py  $CONFIG $CHECKPOINT --img-prefix $I --out-dir $O --launcher pytorch ${@:4}
