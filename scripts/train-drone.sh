CONFIG_FILE=./visdrone/configs/ssd300.py
CHECKPOINT_FILE=work_dirs/ssd300_visdrone/latest.pth
# Please set work dir in config file

python tools/train.py ${CONFIG_FILE} --resume_from $CHECKPOINT_FILE
