CONFIG_FILE=visdrone/configs/ssd300.py
# CHECKPOINT_FILE=zoo/checkpoints/vgg16_caffe-292e1171.pth
CHECKPOINT_FILE=work_dirs/ssd300_visdrone/epoch_12.pth
RESULT_FILE=result.pkl
EVAL_METRICS=bbox

# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS} --show
