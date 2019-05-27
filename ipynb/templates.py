# Get cfg
import mmcv
config_file = 'visdrone/configs/ssd300.py'
cfg = mmcv.Config.fromfile(config_file)


# Get model
from mmdet.models import build_detector
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
