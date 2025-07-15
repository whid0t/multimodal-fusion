_base_ = [
    '../_base_/models/deeplabv3_r18-d8_wildscenes.py',
    '../_base_/default_runtime_wildscenes.py', '../_base_/schedules/schedule_80k_wildscenes.py'
]

# Normalization config for single GPU training
norm_cfg = dict(type='BN', requires_grad=True)

# Override model settings to match 3D training and add OHEM sampler
model = dict(
    decode_head=dict(
        num_classes=13,  # Only using 13 classes to match the classes available to the 3D model
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)), 
    auxiliary_head=dict(
        num_classes=13,  # same as decode_head
        norm_cfg=norm_cfg),
    backbone=dict(norm_cfg=norm_cfg)
)

# Training and testing configurations
train_cfg = dict()
test_cfg = dict(mode='whole')

# Override data configuration to ensure correct paths
# After running the setup_data.py script, the 2D data should be located in .../processed/wildscenes_opt2d/<split>/image and .../processed/wildscenes_opt2d/<split>/indexLabel
data_root = '/home/s2955369/DeepViewAgg/dataset/WildScenes/data/processed/wildscenes_opt2d'

# EXACT 3D class order (13 classes)
classes = [
    'bush',           # 0 - exactly same as 3D
    'dirt',           # 1 - exactly same as 3D
    'fence',          # 2 - exactly same as 3D
    'grass',          # 3 - exactly same as 3D
    'gravel',         # 4 - exactly same as 3D
    'log',            # 5 - exactly same as 3D
    'mud',            # 6 - exactly same as 3D
    'object',         # 7 - exactly same as 3D
    'other-terrain',  # 8 - exactly same as 3D
    'rock',           # 9 - exactly same as 3D
    'structure',      # 10 - exactly same as 3D
    'tree-foliage',   # 11 - exactly same as 3D
    'tree-trunk',     # 12 - exactly same as 3D
]

# EXACT 3D palette (same RGB values as 3D model)
palette = [
    [230, 25, 75],     # bush - from 3D METAINFO
    [60, 180, 75],     # dirt - from 3D METAINFO
    [0, 128, 128],     # fence - from 3D METAINFO
    [128, 128, 128],   # grass - from 3D METAINFO
    [145, 30, 180],    # gravel - from 3D METAINFO
    [128, 128, 0],     # log - from 3D METAINFO
    [255, 225, 25],    # mud - from 3D METAINFO
    [250, 190, 190],   # object - from 3D METAINFO
    [70, 240, 240],    # other-terrain - from 3D METAINFO
    [170, 255, 195],   # rock - from 3D METAINFO
    [170, 110, 40],    # structure - from 3D METAINFO
    [210, 245, 60],    # tree-foliage - from 3D METAINFO
    [240, 50, 230],    # tree-trunk - from 3D METAINFO
]

# Label remapping (19 original classes → 13 training classes). 
# This resembles the custom_label_map defined in .../wildscenes/configs/_base_/datasets/wildscenes_standard.py
# Original class IDs are befined in METAINFO in .../wildscenes/mmseg_wildscenes/dataset/wildscenes.py
label_map = {
    0: 255,  # unlabelled → ignored
    1: 8,    # asphalt/concrete → other-terrain
    2: 1,    # dirt → dirt
    3: 6,    # mud → mud
    4: 255,  # water → ignored (same as 3D)
    5: 4,    # gravel → gravel
    6: 8,    # other-terrain → other-terrain
    7: 12,   # tree-trunk → tree-trunk
    8: 11,   # tree-foliage → tree-foliage
    9: 0,    # bush → bush
    10: 2,   # fence → fence
    11: 10,  # other-structure → structure
    12: 7,   # pole → object
    13: 255,   # vehicle → ignored
    14: 9,   # rock → rock
    15: 5,   # log → log
    16: 7,   # other-object → object
    17: 255, # sky → ignored (same as 3D)
    18: 3,   # grass → grass
}

# Image normalization config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LabelRemap', label_map=label_map),  # Custom label remapping
    dict(type='Resize', img_scale=(2016, 1512), ratio_range=(0.5, 2.0)),
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
        img_scale=(2016, 1512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# The val_pipeline should be the same as the test_pipeline, as evaluation
# in this version of MMSegmentation does not use annotations from the pipeline.
# Remapping is handled by passing `label_map` to the custom eval dataset.
val_pipeline = test_pipeline

data = dict(
    samples_per_gpu=2, # batch size that fits in A40
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_root=data_root,
        img_dir='train/image',
        ann_dir='train/indexLabel',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette),
    val=dict(
        type='CustomDatasetEval',  # Use new evaluation dataset
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/indexLabel',  # Still needed for `evaluate` method
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=val_pipeline,  # Use the test pipeline
        classes=classes,
        palette=palette,
        label_map=label_map),  # Pass the label_map for correctmetric calculation
    test=dict(
        type='CustomDatasetEval',
        data_root=data_root,
        img_dir='test/image',
        ann_dir='test/indexLabel',
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        label_map=label_map)
)

# Custom work directory
work_dir = './work_dirs/deeplabv3_r18-d8_512x512_80k_wildscenes_aligned'

evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)

# Optimizer based on the one from KITTI-360
optimizer = dict(type='SGD',
                 lr=1e-3,                    # 0.01*(2/20), since can't use GradientCumulativeOptimizerHook
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(custom_keys={'decode_head': dict(lr_mult=10.)}))