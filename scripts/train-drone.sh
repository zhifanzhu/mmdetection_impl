MMDET=/home/damon/Github/mmlab/mmdetection
export PYTHONPATH=$PYTHONPATH:$MMDET

CURDIR=`dirname "$0"`
CURDIR=`realpath $CURDIR`
CONFIG_FILE=`ls $CURDIR/*.py`
# Please set work dir in config file

CHECKPOINT_FILE=$CURDIR/latest.pth
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "train-drone.sh: Found checkpoint file"
    cd $MMDET
    python $MMDET/tools/train.py ${CONFIG_FILE} --resume_from $CHECKPOINT_FILE
else
    echo "train-drone.sh: No checkpoint, fresh start"
    cd $MMDET
    python $MMDET/tools/train.py ${CONFIG_FILE}
fi
