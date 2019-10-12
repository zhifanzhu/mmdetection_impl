# model settings
input_size = 300
model = dict(
    type='SeqSingleStageDetector',
    pretrained='zoo/mobilenet_v2.pth.tar',
    backbone=dict(
        type='SSDMobileNetV2',
        input_size=input_size,
        frozen_stages=-1,
        norm_eval=True,
    ),
    neck=None,
    temporal_module=dict(
        type='Identity'),
    bbox_head=dict(
        type='SSDLiteHead',
        input_size=input_size,
        norm_eval=True,
        in_channels=(576, 1280, 512, 256, 256, 128),
        num_classes=31,
        anchor_strides=(16, 32, 64, 128, 150, 300),
        basesize_ratio_range=(0.2, 0.9),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2)))
# model training and testing settings
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# dataset settings
train_ds_type = 'FixFramesVIDDataset'
dataset_type = 'SeqVIDDataset'
data_root = 'data/ILSVRC2015/'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, skip_img_without_anno=False),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=[
        dict(
            type='FixFramesVIDDataset',
            seq_len=8,
            ann_file=data_root + 'ImageSets/VID/VID_train_15frames.txt',
            img_prefix=data_root,
            pipeline=train_pipeline),
        dict(
            type='SeqDET30Dataset',
            divisor=32,
            seq_len=8,
            ann_file=data_root + 'ImageSets/VID/DET_train_30classes.txt',
            img_prefix=data_root,
            pipeline=train_pipeline),
    ],
    val=dict(
        type=dataset_type,
        seq_len=24,
        ann_file=data_root + 'ImageSets/VID/VID_val_videos_mini.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        seq_len=24,
        ann_file=data_root + 'ImageSets/VID/VID_val_videos.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1, num_evals=1000, shuffle=True)
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './workvids/ssd300MV2_iden_ne1_fix15_augdet'
load_from = './zoo/SSDMobileV2DetVidEpoch24.pth'
resume_from = None
workflow = [('train', 1)]
