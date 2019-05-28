MMDET=/home/damon/Github/mmlab/mmdetection
export PYTHONPATH=$PYTHONPATH:$MMDET

CURDIR=`dirname "$0"`
CURDIR=`realpath $CURDIR`
CONFIG_FILE=`ls $CURDIR/*.py`

CHECKPOINT_FILE=$CURDIR/latest.pth
RESULT_FILE=$CURDIR/result.pkl
EVAL_METRICS=bbox

# single-gpu testing
cd $MMDET
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS}
# python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS} --show
