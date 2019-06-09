# Step one: install mmdetection
conda create -n open-mmlab python=3.7 -y
source activate open-mmlab

# conda install -c pytorch pytorch torchvision -y
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install cython -y
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
./compile.sh
# mmcv will be installed here
pip install -e .  # "pip install ." for installation mode,

# Step two: install local mmcv
cd /path/to/parent
git clone zhuzhifan@202.119.84.34:~/repositories/mmcv.git
cd mmcv
git checkout -b dev origin/dev
pip install -e .
cd mmdetection

# Step three: link datasets
mkdir data
# ln -s $COCO_ROOT data
ln -sfn /path/to/work_dirs word_dirs
ln -sfn /path/to/zoo zoo

# Tensorboard
pip install tb-nightly future


# NOTE BEFORE YOU RUN
# check following fields:
#   pretrained/loaded_from, NOTE:  for trained backbone, use pretrained; for trained detector, use loaded_from, distinguish between this two!!!
#   imgs_per_gpu
#   learing-rate,
#   num_classes
#   interval,
#   work_dir,
#   workflow
#   test_cfg.max_per_img
#   dataset and ann files
#  test_mode : 'train' False by default, 'val' and 'test' : True
#  tensorboard
