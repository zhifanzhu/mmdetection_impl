# New iden (fixframe)
# model settings
model = dict(
    type='SeqRetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    temporal_module=dict(
        type='Identity'),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=31,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
train_ds_type = 'FixFramesVIDDataset'
dataset_type = 'SeqVIDDataset'
data_root = 'data/ILSVRC2015/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, skip_img_without_anno=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
seqdet_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, skip_img_without_anno=False),
    # dict(
    #     type='SeqPhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    # dict(
    #     type='SeqExpand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     ratio_range=(1, 4)),
    # dict(
    #     type='SeqMinIoURandomCrop',
    #     min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
    #     min_crop_size=0.3),
    # dict(
    #     type='RandomRatioCrop',
    #     min_crop_ratio=(0.7, 0.7),
    #     max_crop_ratio=(0.9, 0.9)),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SeqRandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=[
        dict(
            type='SeqVIDDataset',
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
            pipeline=seqdet_train_pipeline),
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
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
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
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './workvids/retinaR50_iden_fix15withdet'
load_from = './zoo/RetinaR50DetVidEpoch20.pth'
resume_from = None
workflow = [('train', 1)]
