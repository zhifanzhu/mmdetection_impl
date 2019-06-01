# mmdet import
import mmdet
import mmcv
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from mmdet.datasets import get_dataset
from mmdet.datasets import transforms

# Get cfg
import mmcv
config_file = 'visdrone/configs/ssd300.py'
cfg = mmcv.Config.fromfile(config_file)


# Get model
from mmdet.models import build_detector
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)


