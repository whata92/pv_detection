_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

dataset_type = 'OriginalDataset'
data_root = 'data/processed'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img',
        ann_dir='annotation',
        pipeline=train_pipeline,
        split='/workspace/data/processed/train.txt'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img',
        ann_dir='annotation',
        pipeline=test_pipeline,
        split='/workspace/data/processed/val.txt'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img',
        ann_dir='annotation',
        pipeline=test_pipeline,
        split='/workspace/data/processed/val.txt'
    )
)

runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=50)
evaluation = dict(interval=50, metric='mIoU', pre_eval=True)