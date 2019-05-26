CONFIG_FILE=visdrone/configs/ssd300.py
CHECKPOINT_FILE=work_dirs/ssd300_visdrone/latest.pth
RESULT_FILE=result.pkl
EVAL_METRICS=bbox

# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS}
# python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval ${EVAL_METRICS} --show
