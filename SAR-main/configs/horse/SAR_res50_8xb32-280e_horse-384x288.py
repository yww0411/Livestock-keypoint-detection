_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(imports='SAR')

# runtime
train_cfg = dict(max_epochs=280, val_interval=70)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=280,
        milestones=[240, 270],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmapCoord', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    head=dict(
        type='SARHead',
        in_channels=2048,
        out_channels=18,
        with_heatmap=True,
        codec=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data_process/horse/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]



# data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train_annotations.coco.json',
        data_prefix=dict(img='images/train/'),
        # 指定元信息配置文件
        metainfo=dict(from_file='configs/_base_/datasets/custom4horse.py'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val_annotations.coco.json',
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        # use_gt_bbox=True,
        # bbox_file='',
        data_prefix=dict(img='images/val/'),
        # 指定元信息配置文件
        metainfo=dict(from_file='configs/_base_/datasets/custom4horse.py'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# test_dataloader = val_dataloader

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/test_annotations.coco.json',
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        # use_gt_bbox=True,
        # bbox_file='',
        data_prefix=dict(img='images/test/'),
        # 指定元信息配置文件
        metainfo=dict(from_file='configs/_base_/datasets/custom4horse.py'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# evaluators
val_evaluator = [
    dict(type='CocoMetric',ann_file=data_root + 'annotations/val_annotations.coco.json'),
]

# test_evaluator = val_evaluator

test_evaluator = [
    dict(type='CocoMetric',ann_file=data_root + 'annotations/test_annotations.coco.json'),
]
