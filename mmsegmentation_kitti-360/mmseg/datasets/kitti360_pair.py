# mmseg/datasets/kitti360_pair.py
#
# Reads an ann_file in which **each line is**
#     <relative/path/to/img> <relative/path/to/gt>
# relative to `data_root`.
#
from .custom import CustomDataset
from .builder import DATASETS
import os
import numpy as np

@DATASETS.register_module()
class Kitti360PairDataset(CustomDataset):
    """KITTI-360 semantic segmentation dataset with <img gt> pairs in ann_file provided by the official KITTI-360 dataset"""

    CLASSES = (
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

    PALETTE = [  # Cityscapes colours
        [128,  64,128],[244, 35,232],[ 70, 70,70],[102,102,156],[190,153,153],
        [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],
        [ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],
        [  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]
    ]
    
    # label-ID → train-ID  (only the 19 valid classes)
    # full_map = {i: 255 for i in range(256)}           # default to ignore
    # full_map.update({                                 # overwrite valid labels
    #     7: 0,  8: 1, 11: 2, 12: 3, 13: 4,
    #     17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    #     23:10, 24:11, 25:12, 26:13, 27:14,
    #     28:15, 31:16, 32:17, 33:18})

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_dir='',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        self.ann_file = ann_file
        # self.ann_dir   = self.data_root
        
        super().__init__(
            img_dir=img_dir,
            pipeline=pipeline,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        self.label_map = None
        self.full_map = None

    def load_annotations(self, *args, **kwargs):
        """Parse <image  label> pairs from self.ann_file.

        *args / **kwargs absorb the unused positional arguments that mmseg passes
        (img_dir, img_suffix, seg_map_suffix, split, etc.).
        """
        img_infos = []
        txt_path = os.path.join(self.data_root, self.ann_file)
        with open(txt_path) as f:
            for line in f:
                img_rel, seg_rel = line.strip().split()
                img_infos.append(
                    dict(
                        filename=os.path.join(self.data_root, img_rel),
                        ann=dict(seg_map=os.path.join(self.data_root, seg_rel),
                                 seg_fields=['gt_semantic_seg'])))
        return img_infos
