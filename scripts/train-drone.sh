MMDET=$HOME/Github/mmlab/mmdetection
export PYTHONPATH=$PYTHONPATH:$MMDET

CURDIR=`dirname "$0"`
cd $CURDIR
# CURDIR=`realpath $CURDIR`
CURDIR=$PWD
CONFIG_FILE=`ls $CURDIR/*.py`
LOGFILE=$CURDIR/log.txt
WORK_DIR=$CURDIR

CHECKPOINT_FILE=$CURDIR/latest.pth
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "train-drone.sh: Found checkpoint file"
    cd $MMDET
    python $MMDET/tools/train.py ${CONFIG_FILE} \
        --work_dir $WORK_DIR \
        --resume_from $CHECKPOINT_FILE \
        --validate 2>&1 | tee -a $LOGFILE
else
    echo "train-drone.sh: No checkpoint, fresh start"
    cd $MMDET
    python $MMDET/tools/train.py ${CONFIG_FILE} \
        --work_dir $WORK_DIR \
        --validate 2>&1 | tee -a $LOGFILE
fi
