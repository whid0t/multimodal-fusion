IGNORE_LABEL: int = 255 # Edited to match the label map for WildScenes in the 2D only model, original was -1

from .shapenet import ShapeNet, ShapeNetDataset
from .s3dis import S3DISFusedDataset, S3DIS1x1Dataset, S3DISOriginalFused, S3DISSphere
from .scannet import ScannetDataset, Scannet
from .kitti360 import KITTI360Dataset
