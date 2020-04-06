import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .pair_base import PairBaseDetector


class UpdateNet(nn.Module):
    def __init__(self, in_channels, out_channels, loss_weight=0.1):
        super(UpdateNet, self).__init__()
        self.loss_weight = loss_weight
        self.score_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=2,
                kernel_size=3,
                padding=1,
                stride=1,
            )
        )

        self.trans_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
            )
        )

    def init_weights(self):
        def _init_conv(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                normal_init(m, std=0.01)
        self.apply(_init_conv)
        nn.init.constant_(self.score_conv[-1].bias[0], 1.0)
        nn.init.constant_(self.score_conv[-1].bias[1], 0.0)

    def forward(self, x, x_ref, is_train):
        cat = torch.cat([x, x_ref], dim=1)
        trans = self.trans_conv(cat)
        score = self.score_conv(cat)
        score = torch.softmax(score, dim=1)
        w0 = score[:, 0, :, :]
        out = w0.unsqueeze(1) * x + score[:, 1, :, :].unsqueeze(1) * trans

        if is_train:
            w0_valid = w0[w0 > 0.5]
            term = torch.mean(w0_valid * w0_valid)
            loss = self.loss_weight * term
            return out, loss
        else:
            return out


@DETECTORS.register_module
class PairX1SingleStageDetector(PairBaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PairX1SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.update_nets = nn.ModuleList()
        for _ in range(5):  # 5 is fpn level
            self.update_nets.append(
                UpdateNet(in_channels=256, out_channels=256))
        self.init_weights(pretrained=pretrained)

        # memory cache for testing
        self.prev_memory = None

    def init_weights(self, pretrained=None):
        super(PairX1SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        for l in range(len(self.update_nets)):
            self.update_nets[l].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      ref_img,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        x_ref = self.extract_feat(ref_img)
        x_update = []
        update_losses = []
        for l, net in enumerate(self.update_nets):
            update_feat, _loss = net(x[l], x_ref[l], is_train=True)
            x_update.append(update_feat)
            update_losses.append(_loss)
        update_loss = sum(update_losses) / 5  # 5 = len(self.update_nets)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        losses.update(dict(loss_update=update_loss))
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        x_cache = x
        is_first = img_meta[0]['is_first']
        if is_first:
            x_update = x
        else:
            x_update = []
            for l, net in enumerate(self.update_nets):
                update_feat = net(x[l], self.prev_memory[l], is_train=False)
                x_update.append(update_feat)

        self.prev_memory = x_cache

        outs = self.bbox_head(x_update)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
