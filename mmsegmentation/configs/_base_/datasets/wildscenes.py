# dataset settings
dataset_type = 'CustomDataset'
# WildScenes 2D data path - corrected based on actual location
data_root = '/home/s2955369/DeepViewAgg/dataset/WildScenes/data/processed/wildscenes_opt2d'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2016, 1512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
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

# WildScenes class mapping (15 classes total)
classes = [
    'unlabelled',      # 0
    'tree-trunk',      # 1  
    'tree-foliage',    # 2
    'bush',            # 3
    'grass',           # 4
    'dirt',            # 5
    'gravel',          # 6
    'water',           # 7
    'rock',            # 8
    'log',             # 9
    'sky',             # 10
    'structure',       # 11
    'object',          # 12
    'other-terrain',   # 13
    'fence'            # 14
]

# Color palette for visualization (matches WildScenes semantics)
palette = [
    [0, 0, 0],         # unlabelled (black)
    [139, 69, 19],     # tree-trunk (saddle brown)
    [34, 139, 34],     # tree-foliage (forest green)
    [0, 128, 0],       # bush (green)
    [124, 252, 0],     # grass (lawn green)
    [160, 82, 45],     # dirt (saddle brown)
    [169, 169, 169],   # gravel (dark gray)
    [0, 191, 255],     # water (deep sky blue)
    [105, 105, 105],   # rock (dim gray)
    [210, 180, 140],   # log (tan)
    [135, 206, 235],   # sky (sky blue)
    [255, 165, 0],     # structure (orange)
    [255, 20, 147],    # object (deep pink)
    [205, 133, 63],    # other-terrain (peru)
    [255, 255, 0]      # fence (yellow)
]

data = dict(
    samples_per_gpu=2,  # Batch size per GPU (conservative for older PyTorch)
    workers_per_gpu=2,  # Number of workers (conservative)
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/image',
        ann_dir='train/indexLabel',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/image',
        ann_dir='val/indexLabel',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/image',
        ann_dir='test/indexLabel',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette))
