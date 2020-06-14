from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .registry import DATASETS


@DATASETS.register_module
class TrackDET30Dataset(Dataset):

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 'n02924116',
               'n02958343', 'n02402425', 'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165', 'n01674464', 'n02484322',
               'n03790512', 'n02324045', 'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049',)
    DATASET_NAME = 'vid'

    def __init__(self,
                 **kwargs):
        super(TrackDET30Dataset, self).__init__(**kwargs)

    def prepare_train_img(self, idx):
        """ Pipelines same as PairDET30Datase. """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)

        results['ref_img'] = DC(results['img'].data.clone(), stack=True)
        results['ref_img_metas'] = results['img_metas']  # Is it read-only ?
        results['ref_bboxes'] = DC(results['gt_bboxes'].data.clone(), stack=True)
        results['ref_labels'] = DC(results['gt_labels'].data.clone(), stack=True)

        # We generate 'trackids' field manually
        num_gts = len(results['gt_labels'].data)
        _data = [i + 1 for i in range(num_gts)]  # note we leave 0 for negative/background
        trackids = results['gt_labels'].new_tensor(_data)
        results['ref_trackids'] = DC(trackids, stack=True)

        if len(results['gt_bboxes'].data) == 0:
            return None
        return results


