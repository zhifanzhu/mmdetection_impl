CONFIG_FILE=./configs/ssd300_coco.py
CHECKPOINT_FILE=zoo/vgg16_caffe-292e1171.pth
RESULT_FILE=result.pkl
EVAL_METRICS=bbox

# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS}
