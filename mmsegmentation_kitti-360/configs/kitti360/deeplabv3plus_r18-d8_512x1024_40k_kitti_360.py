# ------------------------------------------------------------
# Fine-tune DeepLab V3+ R-18-D8 on KITTI-360 (80 k iters)
# ------------------------------------------------------------
# mmsegmentation v0.14.1  •  torch 1.7.1  •  CUDA 11.0
# ------------------------------------------------------------

custom_imports = dict(
    imports=['mmseg.datasets.kitti360_pair'],   #  path is resolved relative to cwd/PYTHONPATH
    allow_failed_imports=False)

# ---- 1. BASE MODEL -----------------------------------------------------
norm_cfg = dict(type='SyncBN', requires_grad=True)   # keep Cityscapes defaults, but refer to a single GPU
_base_ = ['../_base_/default_runtime.py']

model = dict(
    type='EncoderDecoder',
    pretrained='torchvision://resnet18',         # ignored, we load_from
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(                                # DeepLabV3+ head
        type='DepthwiseSeparableASPPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 12, 24, 36),
        c1_in_channels=64,
        c1_channels=32,
        dropout_ratio=0.1,
        num_classes=19,                              # ← KITTI-360 / Cityscapes
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(                             # same as upstream
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# ---- 2. DATASET --------------------------------------------------------
# We read explicit <image_path> <label_path> pairs from two text files.
# Each line in the files is missing the fixed prefix
#   "/home/jovyan/DeepViewAgg/dataset/2d/"
# so we point `data_root` to that prefix and let mmsegmentation concatenate.

dataset_type = 'Kitti360PairDataset'
data_root = '/home/jovyan/DeepViewAgg/dataset/2d' # This also includes the common prefix

classes = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle')
palette = [  # Cityscapes palette (19 colours)
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32]
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (376, 736)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize',
         img_scale=(1408, 376),
         ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop',
         crop_size=crop_size,
         cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad',
         size=crop_size,
         pad_val=0,
         seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
    samples_per_gpu=2, # 4 Ran into OOM with an A10 with 23GB
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='2013_05_28_drive_train_frames.txt',
        ann_dir=data_root,
        pipeline=train_pipeline,
        reduce_zero_label=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='2013_05_28_drive_val_frames.txt',
        ann_dir=data_root,
        pipeline=test_pipeline,
        reduce_zero_label=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='2013_05_28_drive_val_frames.txt',
        ann_dir=data_root,
        pipeline=test_pipeline,
        reduce_zero_label=False))

# ---- 3. OPTIMISER & LR -------------------------------------------------

optimizer = dict(type='SGD',
                 lr=5e-3,                 # 0.005
                 momentum=0.9,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=4)
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=0.9,
                 min_lr=1e-6,
                 by_epoch=False)

# ---- 4. RUNNER: 30 000 ITERATIONS -------------------------------------

runner = dict(type='IterBasedRunner', max_iters=40000) # 1/2 of the original training
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(
    interval=10000,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
)

# ---- 5. FP16  -----------------

fp16 = dict(loss_scale='dynamic')

# ---- 6. RUNTIME --------------------------------------------------------

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

log_level = 'INFO'

# run one training phase per epoch/iter; no separate val phase here
workflow = [('train', 1)]

load_from = '/home/jovyan/2d/mmsegmentation/checkpoints/' \
            'deeplabv3plus_r18b-d8_512x1024_80k_cityscapes_20201226_090828-e451abd9.pth'

work_dir = './work_dirs/deeplabv3plus_r18-d8_k360_40k_normal'
