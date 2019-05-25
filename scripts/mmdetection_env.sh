conda create -n open-mmlab python=3.7 -y
source activate open-mmlab

conda install -c pytorch pytorch torchvision -y
conda install cython -y
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
./compile.sh
# mmcv will be installed here
pip install -e .  # "pip install ." for installation mode,

mkdir data
# ln -s $COCO_ROOT data
