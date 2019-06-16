#!/usr/bin/env bash
# Combined dist_train.sh and train-drone.sh

MMDET=$HOME/Github/mmlab/mmdetection
export PYTHONPATH=$PYTHONPATH:$MMDET

PYTHON=${PYTHON:-"python"}

GPUS=$1  # CONFIG=$1

CURDIR=`dirname "$0"`
cd $CURDIR
CURDIR=$PWD
CONFIG=`ls $CURDIR/*.py`
LOGFILE=$CURDIR/log.txt
WORK_DIR=$CURDIR
CHECKPOINT_FILE=$CURDIR/latest.pth

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "train-drone.sh: Found checkpoint file"
    cd $MMDET
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
        $MMDET/tools/train.py $CONFIG --launcher pytorch \
        --work_dir $WORK_DIR \
        --resume_from $CHECKPOINT_FILE \
        --validate 2>&1 | tee -a $LOGFILE
else
    echo "train-drone.sh: No checkpoint, fresh start"
    cd $MMDET
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
        $MMDET/tools/train.py $CONFIG --launcher pytorch \
        --work_dir $WORK_DIR \
        --validate 2>&1 | tee -a $LOGFILE
fi
