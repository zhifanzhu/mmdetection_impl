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
cd /path/to/mmcv  # cd ../mmcv
git clone zhuzhifan@202.119.84.34:~/repositories/mmcv.git
git checkout -b dev origin/dev
pip install -e .
cd /path/to/mmdetection

# Step three: link datasets
mkdir data
# ln -s $COCO_ROOT data
ln -sfn /path/to/work_dirs word_dirs
ln -sfn /path/to/zoo zoo
