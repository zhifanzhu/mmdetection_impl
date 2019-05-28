CFG=work_dirs/ssd300_visdrone/ssd300.py
CKPT=work_dirs/ssd300_visdrone/latest.pth
PFX=/home/damon/DATASETS/Drone2019/VisDrone2019-DET/VisDrone2018-DET-val
ODIR=/tmp/outres

python visdrone/utils/output_to_txt.py --config=$CFG \
    --checkpoint=$CKPT \
    --img-prefix=$PFX \
    --out-dir=$ODIR
