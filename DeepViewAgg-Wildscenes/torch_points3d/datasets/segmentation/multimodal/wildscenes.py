'''
THIS CLASS IS BASED ON THE CONFIGURATION CLASS FOR KITTI-360

This module provides PyTorch Dataset classes for the WildScenes dataset,
specifically tailored for multimodal 3D semantic segmentation tasks.
It handles the loading and preprocessing of 3D LiDAR point clouds and
associated 2D camera images, along with their respective labels and
calibration data.

The core components are:
- `WildScenesCylinderMM`: A `torch_geometric.data.Dataset` implementation
  that loads individual samples (point cloud, image(s), labels, calibration)
  for a given data split (train, val, test). It manages the complexities of
  linking 3D point cloud data with corresponding 2D image data, which might
  involve looking up pre-processed symlinks or deriving paths based on
  timestamps and sequence information.
- `WildScenesDatasetMM`: A wrapper class that instantiates `WildScenesCylinderMM`
  for the train, validation, and test sets, configured via an OmegaConf object.

The module expects a specific directory structure for the WildScenes dataset
and its preprocessed components (e.g., pickle files containing metadata,
symlinked image directories). It also defines a mapping (`RAW2TRAIN`) from
raw WildScenes class IDs to contiguous training IDs.
'''
import os
import numpy as np
import torch
import cv2
import open3d as o3d
import yaml
from torch_geometric.data import Data
import pickle
from torch_points3d.core.data_transform.multimodal.image import DropImagesOutsideDataBoundingBox, PickKImages
# from torch_points3d.datasets.segmentation.kitti360 import KITTI360Sampler
from torch_points3d.core.multimodal.data import MMData
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM
from torch_geometric.data import Dataset
from omegaconf import OmegaConf
from pathlib import Path
from torch_points3d.core.data_transform.multimodal.image import SameSettingImageData, ImageData
from scipy.spatial.transform import Rotation as R
from torch_points3d.utils.multimodal import MAPPING_KEY


# --------------------------- utility -------------------------------- #
# Official 13-class mapping for WildScenes, ignoring unlabeled, water, and sky.
RAW2TRAIN = {
    255:255,  # unlabelled
    0:0,    # bush
    1:1,    # dirt
    2:2,    # fence
    3:3,    # grass
    4:4,    # gravel
    5:5,    # log
    6:6,    # mud
    7:7,    # object (pole)
    8:8,    # other-terrain
    9:9,    # rock
    10:255,  # sky → ignore
    11:10,   # other-structure → structure
    12:11,   # tree-foliage
    13:12,   # tree-trunk
    14:255   # water → ignore
}
NUM_CLASSES = 13


class WildScenesCylinderMM(Dataset):
    '''
    A PyTorch Geometric Dataset class for loading and processing individual
    samples from a split (train, val, test) of the WildScenes multimodal dataset.

    This class is responsible for:
    - Loading 3D point cloud data and their semantic labels from files specified
      in a metadata pickle.
    - Locating and loading corresponding 2D camera images and calibration files.
      It first attempts to find images via a pre-processed 2D data structure
      (typically containing symlinks to original images, e.g., `wildscenes_opt2d`).
      If an image is not found in the pre-processed structure, it falls back to
      deriving the image path from the point cloud\'s sequence and timestamp.
    - Applying 3D and 2D/multimodal transformations.
    - Assembling the data into `torch_points3d.core.multimodal.data.MMData` objects.

    Attributes:
        root (str): The root directory of the WildScenes dataset.
        split (str): The current dataset split (e.g., "train", "val", "test").
        split_pickle_3d_dir (str): Path to the directory containing the 3D metadata
            pickle files (e.g., `wildscenes_infos_train.pkl`). These files list
            point cloud paths, label paths, and sample IDs.
        split_pickle_2d_dir (str, optional): Path to the directory containing the
            pre-processed 2D data structure (e.g., `wildscenes_opt2d`). This
            directory is expected to have subdirectories for each split, containing
            symlinks to the actual image files, organized in an MMSegmentation-like
            structure.
        image_size (tuple): Target (width, height) for resizing loaded images.
        pre_transform_image (callable, optional): Pre-transformations to apply to
            the image data before multimodal transforms.
        transform_image (callable, optional): Transformations to apply to the
            image data, often involving multimodal aspects.
        meta_3d (dict): Loaded metadata from the 3D pickle file for the current split.
        _image_calib_data_from_2d_pickle (dict): A dictionary mapping `sample_id`
            to image and calibration paths, populated from the `split_pickle_2d_dir`.
        _cloud_files (list): List of `Path` objects to point cloud files.
        _label_files (list): List of `Path` objects to 3D label files.
        _image_files (list): List of `Path` objects to image files.
        _calib_files (list): List of `Path` objects to camera calibration files.
    '''
    
    def __init__(self, root, split="train", split_pickle_3d_dir=None, split_pickle_2d_dir=None,
                 transform=None, pre_transform=None,
                 pre_transform_image=None, transform_image=None,
                 image_size=(2016, 1512)):
        super().__init__(root, transform, pre_transform)
        
        if split_pickle_3d_dir is None:
            raise ValueError("split_pickle_3d_dir must be provided")
        
        print(f"[PATH_DEBUG] Dataset root: {root}")
        print(f"[PATH_DEBUG] 2D data directory: {split_pickle_2d_dir}")
        
        self.split_pickle_3d_dir = Path(split_pickle_3d_dir)
        self.split_pickle_2d_dir = Path(split_pickle_2d_dir) if split_pickle_2d_dir else None
        self.root = root
        self.split = split
        self.image_size = image_size

        self.pre_transform_image = pre_transform_image
        self.transform_image = transform_image

        # Load and filter data to include only samples with valid images
        self._length = self._load_split_with_filtering(split)

    def _load_split_with_filtering(self, split):
        """Load and filter split data to only include samples with valid images."""
        print(f"[INFO] Loading {split} split data and filtering for valid images...")
        
        # Load 3D metadata (point clouds, 3D labels)
        pkl_3d_path = self.split_pickle_3d_dir / f"wildscenes_infos_{split}.pkl"
        if not pkl_3d_path.exists():
            raise FileNotFoundError(f"3D pickle file not found: {pkl_3d_path}")
        with open(pkl_3d_path, "rb") as f:
            meta_3d = pickle.load(f)       # list of dicts
        
        # Load 2D metadata from directory structure
        image_calib_data_from_2d_pickle = {}
        if self.split_pickle_2d_dir:
            split_2d_path = self.split_pickle_2d_dir / split / "image"
            if split_2d_path.exists():
                try:
                    print(f"[PATH_DEBUG] Loading 2D data from: {split_2d_path}")
                    # Get all image symlinks
                    for img_link in split_2d_path.glob("*.png"):
                        # Get the real path the symlink points to
                        real_path = img_link.resolve()
                        # Extract sample_id from filename
                        sample_id = img_link.stem  # remove extension
                        
                        # Print only first 2 entries to avoid too much printing
                        if len(image_calib_data_from_2d_pickle) < 2:
                            print(f"[PATH_DEBUG] Symlink: {img_link}")
                            print(f"[PATH_DEBUG] Resolves to: {real_path}")
                        
                        # Get sequence from real path
                        seq = real_path.parent.parent.name  # e.g. K-03
                        calib_path = Path(self.root).parent / "WildScenes2d" / seq / "camera_calibration.yaml"
                        
                        image_calib_data_from_2d_pickle[sample_id] = {
                            "image": img_link,  # Use symlink path instead of real path
                            "calib": calib_path
                        }
                    print(f"[PATH_DEBUG] Total 2D samples loaded: {len(image_calib_data_from_2d_pickle)}")
                except Exception as e:
                    print(f"[PATH_DEBUG] Error loading 2D data: {str(e)}")
            else:
                print(f"[PATH_DEBUG] 2D directory not found: {split_2d_path}")

        # Process and filter samples to include only those with valid images
        frames_3d = meta_3d["data_list"]
        self._cloud_files, self._label_files, self._image_files, self._calib_files = [], [], [], []
        missing_in_2d_pickle_count = 0
        skipped_missing_images = 0
        total_samples = len(frames_3d)
        
        for f_3d in frames_3d:
            cloud = Path(f_3d["lidar_points"]["lidar_path"])
            label = Path(f_3d["pts_semantic_mask_path"])
            sample_id_3d = f_3d["sample_id"]

            # Convert PLY path to BIN path if needed
            if cloud.suffix == ".ply":
                cloud = self.convert_ply_to_bin_path(cloud)
            
            # Try to get image and calib paths from the 2D pickle first
            paths_from_2d = image_calib_data_from_2d_pickle.get(sample_id_3d)
            
            if paths_from_2d:
                img = paths_from_2d["image"]
                calib = paths_from_2d["calib"]
            else:
                # Fallback to derive_2d if not found in 2D pickle or 2D pickle wasn't loaded
                if sample_id_3d not in image_calib_data_from_2d_pickle and self.split_pickle_2d_dir:
                     missing_in_2d_pickle_count += 1
                seq = cloud.parent.parent.name          # e.g. 'K-03'
                img, calib = self.derive_2d(seq, sample_id_3d)

            # Check if both point cloud and image files exist before adding to dataset
            if Path(cloud).exists() and Path(img).exists():
                self._cloud_files.append(cloud)
                self._label_files.append(label)
                self._image_files.append(img)
                self._calib_files.append(calib)
            else:
                skipped_missing_images += 1
                if skipped_missing_images <= 5:  # Only print first few for brevity
                    if not Path(cloud).exists():
                        print(f"[INFO] Skipping sample {sample_id_3d}: point cloud not found at {cloud}")
                    if not Path(img).exists():
                        print(f"[INFO] Skipping sample {sample_id_3d}: image not found at {img}")
        
        if missing_in_2d_pickle_count > 0:
            # This is just a warning, since the number of images is different from the number of point clouds it will print every time just to give info about the number of sample differences
            print(f"[WARNING] WildScenesCylinderMM: {missing_in_2d_pickle_count} sample_ids from 3D pickle were not found in the 2D pickle for split '{self.split}'. Used derive_2d fallback for them.")
        
        valid_samples = len(self._cloud_files)
        print(f"[INFO] {split} split: {valid_samples}/{total_samples} samples with valid images ({skipped_missing_images} skipped)")
        print(f"[INFO] WildScenesCylinderMM loaded {len(self._image_files)} image references for split '{self.split}'")
        print(f"[INFO] WildScenesCylinderMM loaded {valid_samples} point clouds for split '{self.split}'")
        
        return valid_samples

    def __len__(self):
        return self._length

    @property
    def raw_file_names(self):
        return self._cloud_files + self._label_files + self._image_files + self._calib_files

    @property
    def processed_file_names(self):
        # In the future, this would list processed data files, for now empty. Data can be seen in the processed folder which will contain only 3 .pt files - one for each split
        return []
    
    @property
    def num_classes(self):
        return NUM_CLASSES

    @property
    def num_node_features(self):
        # We use rgb (3) + pos_z (1) = 4 features.
        return 4

    def process(self):
        print("[INFO] Process called on WildScenesCylinderMM.")

    def get(self, idx):
        # Load your data here
        return {
            "cloud": self._cloud_files[idx],
            "label": self._label_files[idx],
            "image": self._image_files[idx],
            "calibration": self._calib_files[idx]
        }

    def derive_2d(self, seq, sample_id):
        """
        WildScenes opt3d pickle does not store RGB paths.
        Derive the PNG name from sample_id: 'T.sec_nsec' -> 'T-sec-nsec.png'
        
        The 2D data follows MMSegmentation directory structure:
        dataset/WildScenes/data/processed/wildscenes_opt2d/[train|val|test]/image/
        """
        # Convert timestamp format from dots to dashes, since there are differences in the format of the timestamp in the 2D and 3D data
        img_name = f"{sample_id.replace('.', '-')}.png"
        
        # First try to find the image in the MMSegmentation structure
        if self.split_pickle_2d_dir:
            img_path = Path(self.split_pickle_2d_dir) / self.split / "image" / img_name
            if img_path.exists() or img_path.is_symlink():
                # Get sequence from the real path the symlink points to
                real_path = img_path.resolve()
                seq = real_path.parent.parent.name  # e.g. K-03
                calib_path = Path(self.root).parent / "WildScenes2d" / seq / "camera_calibration.yaml"
                print(f"[PATH_DEBUG] Found image in MMSeg structure at {img_path}")
                print(f"[PATH_DEBUG] Using calibration from {calib_path}")
                return img_path, calib_path
        
        # Fallback to the original WildScenes2d structure
        base = Path(self.root).parent / "WildScenes2d" / seq
        img = base / "image" / img_name
        calib = base / "camera_calibration.yaml"
        print(f"[PATH_DEBUG] Falling back to original structure, looking in {img}")
        return img, calib

    def convert_ply_to_bin_path(self, ply_path):
        """
        Convert a .ply path to the corresponding .bin path.
        The .bin files are located in WildScenes3d/<seq>/Clouds/ directory. While the .ply files are located in WildScenes3d/<seq>/FullClouds/ directory.
        
        Args:
            ply_path (Path): Original .ply file path
            
        Returns:
            Path: Corresponding .bin file path
        """
        ply_path = Path(ply_path)
        
        # Extract sequence and filename
        if "WildScenes3d" in str(ply_path):
            # Parse the sequence from the path
            parts = ply_path.parts
            seq_idx = None
            for i, part in enumerate(parts):
                if part == "WildScenes3d":
                    seq_idx = i + 1
                    break
            
            if seq_idx is not None and seq_idx < len(parts):
                seq = parts[seq_idx]  # e.g., "K-01"
                filename = ply_path.stem + ".bin"  # Replace .ply with .bin
                
                # TODO Construct the new path pointing to the Clouds directory, you probably want to change this to the path to the 3D data
                bin_path = Path("/deepstore/datasets/fmt/kitti-360/61541v003/data/WildScenes/WildScenes3d") / seq / "Clouds" / filename
                
                print(f"[PATH_DEBUG] Converting PLY path: {ply_path}")
                print(f"[PATH_DEBUG] To BIN path: {bin_path}")
                
                return bin_path
        
        # If we can't parse the path properly, return the original
        print(f"[WARNING] Could not convert PLY path to BIN path: {ply_path}")
        return ply_path

    def __getitem__(self, idx):
        # Get file paths for this sample
        cloud_file = self._cloud_files[idx]
        label_file = self._label_files[idx] 
        img_path = self._image_files[idx]
        calib_path = self._calib_files[idx]
        
        # Get sample_id from cloud file
        sample_id = cloud_file.stem

        # Load 3D point cloud and features
        if not cloud_file.is_absolute():
            # If the path is relative, make it absolute using the dataset root
            cloud_file = Path(self.root).parent / cloud_file
        
        print(f"[PATH_DEBUG] Loading point cloud from: {cloud_file}")
        print(f"[PATH_DEBUG] File extension: {cloud_file.suffix}")
        
        if cloud_file.suffix == ".bin":           # WildScenes default
            print(f"[PATH_DEBUG] Loading .bin file with shape inference")
            coords = np.fromfile(str(cloud_file), dtype=np.float32)  # Convert Path to string
            print(f"[PATH_DEBUG] Raw data shape: {coords.shape}")
            coords = coords.reshape(-1, 3)        # (N, 3)  XYZ only point clouds
            print(f"[PATH_DEBUG] Reshaped to: {coords.shape}")
        else:                                     # .ply fallback
            print(f"[PATH_DEBUG] Loading .ply file with Open3D")
            pcd = o3d.io.read_point_cloud(str(cloud_file))
            coords = np.asarray(pcd.points, dtype=np.float32)
            print(f"[PATH_DEBUG] PLY file loaded with shape: {coords.shape}")

        if coords.shape[0] == 0:
            raise ValueError(f"[ERROR] No points found in {cloud_file}")

        print(f"[PATH_DEBUG] Successfully loaded {coords.shape[0]} points from {cloud_file}")

        # Store original size for debugging
        original_coords_size = coords.shape[0]

        # INTELLIGENT POINT REDUCTION - Apply before any transforms
        target_points = 200000 # First run was with 50k with bucket_safe, but lost a lot of context
        if coords.shape[0] > target_points:
            print(f"[SAMPLING] REDUCTION NEEDED: {coords.shape[0]} points")
            coords, selected_indices = self.intelligent_point_reduction(
                coords, 
                target_points=target_points, 
                method="voxel_fps"  # The best one
            )
            print(f"[SAMPLING] Reduced from {original_coords_size} to {coords.shape[0]} points")
        else:
            selected_indices = np.arange(coords.shape[0])
            print(f"[SAMPLING] NO REDUCTION NEEDED: {coords.shape[0]} points")

        # Store original size for debugging
        self._original_coords_size = coords.shape[0]

        # Initialize features with RGB and height
        rgb = np.zeros((coords.shape[0], 3), dtype=np.float32)  # RGB initialized to zeros
        pos_z = coords[:, 2:3]  # Height feature
        
        # Create initial features with proper dimensionality for the model
        initial_feat_dim = 4  # Match the model's expected input dimension (RGB + height)
        feats = np.concatenate([rgb, pos_z], axis=1)  # Shape (N, 4)

        # Load labels
        # Use original 3D labels
        if isinstance(label_file, str):
            label_file = Path(label_file)
        # Convert dots to dashes in the filename
        label_stem = label_file.stem
        label_file = label_file.parent / f"{label_stem}.label"
        if not label_file.is_absolute():
            # If the path is relative, make it absolute using the dataset root
            label_file = Path(self.root).parent / label_file
        print(f"[PATH_DEBUG] Loading 3D label from: {label_file}")
        raw = np.fromfile(str(label_file), dtype=np.uint32)
        if raw.size != coords.shape[0]:
            # Need to handle the case where labels don't match reduced points
            if hasattr(self, '_original_coords_size'):
                # If we reduced points, we need to subsample labels too
                raw = raw[selected_indices]
            if raw.size != coords.shape[0]:
                raise ValueError(f"Label/point mismatch: {raw.size} labels vs {coords.shape[0]} points")
        y = np.vectorize(RAW2TRAIN.get)(raw).astype(np.int64)

        # Build torch_geometric.Data
        pos_tensor = torch.tensor(coords, dtype=torch.float32)
        data = Data(
            pos=pos_tensor,
            coords=pos_tensor,  # Add coords attribute for SparseConv3D
            x=torch.tensor(feats, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.int64),
        )
        
        # CRITICAL DEBUG: Log point counts at each stage to keep track of the point reduction and potential overflows
        print(f"[POINT_COUNT_DEBUG] Sample {sample_id}:")
        print(f"  Raw points: {original_coords_size}")
        print(f"  After reduction: {coords.shape[0]}")
        print(f"  Data.pos shape: {data.pos.shape}")
        print(f"  About to apply transforms...")

        # Create mapping features
        # Create mapping features with proper dimensionality for view pooling (padded to 8-dim for pre-trained model)
        mapping_feat_dim = 8  # Pre-trained model expects 8-dim features
        mapping_feats = torch.zeros((coords.shape[0], mapping_feat_dim), dtype=torch.float32)
        mapping_feats[:, 0] = 1.0  # Set first dimension to 1 to indicate valid mapping
        
        # Create mapping indices
        index_map = torch.arange(coords.shape[0], dtype=torch.long)  # simple 1-to-1
        setattr(data, MAPPING_KEY, index_map)  # MAPPING_KEY == 'mapping_index', gotten from the wildscenes.yaml file
        setattr(data, "mapping_features", mapping_feats)  # Add mapping features

        # Load & process image
        # Read the image (we know it exists since we filtered during initialization)
        img_rgb = cv2.imread(str(img_path))
        if img_rgb is None:
            raise ValueError(f"[ERROR] Failed to load image at {img_path} (should have been filtered out)")
        
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (2016, 1512)) # Tested with 1008x756, ideally should be 2016x1512 if memory allows
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_tensor = img_tensor.float() / 255.0  # Normalize to [0, 1]
        
        # Convert RGB image to 4-channel format expected by the model
        img_feat = torch.zeros((1, 4, img_tensor.shape[2], img_tensor.shape[3]), dtype=torch.float32)
        img_feat[:, :3] = img_tensor  # First 3 channels are normalized RGB
        img_feat[:, 3] = 1.0  # Last channel is a validity flag

        # Read camera intrinsics / extrinsics 
        with open(calib_path, "r") as f:
            calib_data = yaml.safe_load(f)

        fx, fy, cx, cy = calib_data["centre-camera"]["intrinsics"]["K"]
        translation     = np.array(calib_data["centre-camera"]["extrinsics"]["translation"],
                                   dtype=np.float32).reshape(3, 1)           # (3,1)
        quat            = np.array(calib_data["centre-camera"]["extrinsics"]["rotation"],
                                   dtype=np.float32)                         # [qx,qy,qz,qw]
        rot_mat         = R.from_quat(quat).as_matrix().astype(np.float32)   # (3,3)

        K = np.array([[fx, 0,  cx],
                      [0,  fy, cy],
                      [0,   0,  1]], dtype=np.float32)                       # (3,3)

        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = rot_mat
        extr[:3, 3:] = translation
        extr = torch.from_numpy(extr).unsqueeze(0)
        
        # Wrap the image so transforms keep it
        img_ss = SameSettingImageData(
            path       = np.array([str(img_path)], dtype=object),
            pos        = torch.zeros(1, 3),  # Add position tensor for 1 view
            opk        = None,
            ref_size   = self.image_size,   # from your dataset_opt
            downscale  = 1,
            rollings   = None,
            crop_size  = None,
            crop_offsets = None,
            x          = img_feat,
            mappings   = None,
            mask       = None,
            visibility = None,
            fx         = torch.tensor([fx], dtype=torch.float32),
            fy         = torch.tensor([fy], dtype=torch.float32),
            mx         = torch.tensor([cx], dtype=torch.float32),
            my         = torch.tensor([cy], dtype=torch.float32),
            extrinsic  = extr,  # Now has shape [1, 4, 4]
        )

        # Group it (num_views == 1)
        img_group = img_ss

        # Apply multimodal image transforms 
        if self.pre_transform_image:
            data, img_group = self.pre_transform_image(data, img_group)
        if self.transform_image:
            data, img_group = self.transform_image(data, img_group)

        # Create multimodal MMData
        calibration = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "translation": translation.reshape(-1).tolist(),
            "rotation": quat.tolist()
        }
        mm_data = MMData(data, image=img_group)
        mm_data.calibration = calibration
        
        return mm_data

    def intelligent_point_reduction(self, coords, target_points=15000, method="bucket_safe"):
        """
        Intelligently reduce point cloud density using various methods.
        
        Args:
            coords (np.ndarray): Point coordinates (N, 3)
            target_points (int): Target number of points
            method (str): Sampling method ("bucket_safe", "voxel_fps", "uniform_grid", "distance_based")
            
        Returns:
            np.ndarray: Reduced point coordinates
            np.ndarray: Indices of selected points
        """
        if coords.shape[0] <= target_points:
            return coords, np.arange(coords.shape[0])
        
        print(f"[SAMPLING] Reducing {coords.shape[0]} points to {target_points} using {method}")
        
        if method == "bucket_safe":
            # Ultra-conservative sampling designed to prevent bucket overflow
            voxel_size = 0.3  # 30cm voxels - much larger than final 5cm
            coords_min = coords.min(axis=0)
            
            # Voxelize points
            voxel_coords = ((coords - coords_min) / voxel_size).astype(np.int32)
            voxel_dict = {}
            for i, vc in enumerate(voxel_coords):
                key = tuple(vc)
                if key not in voxel_dict:
                    voxel_dict[key] = []
                voxel_dict[key].append(i)
            
            # Take only one point per coarse voxel (center point)
            coarse_indices = []
            for indices in voxel_dict.values():
                center_idx = indices[len(indices)//2]  # Take middle point
                coarse_indices.append(center_idx)
            
            coarse_indices = np.array(coarse_indices)
            coarse_coords = coords[coarse_indices]
            
            print(f"[SAMPLING] Coarse voxel reduction: {coords.shape[0]} -> {coarse_coords.shape[0]} points")
            
            # Second: If still too many, use random sampling
            if coarse_coords.shape[0] > target_points:
                random_indices = np.random.choice(coarse_coords.shape[0], target_points, replace=False)
                final_indices = coarse_indices[random_indices]
                print(f"[SAMPLING] Random reduction: {coarse_coords.shape[0]} -> {target_points} points")
                return coords[final_indices], final_indices
            else:
                return coarse_coords, coarse_indices
                
        elif method == "voxel_fps":
            # First: Coarse voxel grid to reduce extreme density
            voxel_size = 0.10  # Before edit: 7cm voxels
            coords_min = coords.min(axis=0)
            coords_max = coords.max(axis=0)
            
            # Voxelize points
            voxel_coords = ((coords - coords_min) / voxel_size).astype(np.int32)
            voxel_dict = {}
            for i, vc in enumerate(voxel_coords):
                key = tuple(vc)
                if key not in voxel_dict:
                    voxel_dict[key] = []
                voxel_dict[key].append(i)
            
            # Take center point from each voxel
            voxel_centers = []
            for indices in voxel_dict.values():
                # center_idx = indices[len(indices)//2]  # Take middle point
                 # take the point farthest from the cell centroid:
                pts = coords[indices]
                centroid = pts.mean(axis=0)
                dists = np.linalg.norm(pts - centroid[None,:], axis=1)
                center_idx = indices[np.argmax(dists)]
                voxel_centers.append(center_idx)
            
            voxel_centers = np.array(voxel_centers)
            reduced_coords = coords[voxel_centers]
            
            print(f"[SAMPLING] Voxel reduction: {coords.shape[0]} -> {reduced_coords.shape[0]} points")
            
            # Second: If still too many, use FPS-like sampling
            if reduced_coords.shape[0] > target_points:
                print(f"[SAMPLING] Reduced points are sitll too much - point count {reduced_coords.shape[0]}. Using FPS to reduce to {target_points} points.")
                # Simple farthest point sampling approximation
                selected_indices = self.farthest_point_sampling_numpy(reduced_coords, target_points)
                final_indices = voxel_centers[selected_indices]
                return coords[final_indices], final_indices
            else:
                return reduced_coords, voxel_centers
                
        elif method == "uniform_grid":
            # Create a uniform 3D grid and sample one point per cell
            grid_size = self.estimate_grid_size(coords, target_points)
            coords_min = coords.min(axis=0)
            
            grid_coords = ((coords - coords_min) / grid_size).astype(np.int32)
            unique_coords, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
            
            print(f"[SAMPLING] Grid reduction: {coords.shape[0]} -> {len(unique_indices)} points")
            return coords[unique_indices], unique_indices
            
        elif method == "distance_based":
            # Sample based on distance from center, preserving structure
            center = coords.mean(axis=0)
            distances = np.linalg.norm(coords - center, axis=1)
            
            # Create distance-based bins and sample uniformly from each
            n_bins = int(np.sqrt(target_points))
            bin_edges = np.linspace(distances.min(), distances.max(), n_bins + 1)
            
            selected_indices = []
            points_per_bin = target_points // n_bins
            
            for i in range(n_bins):
                mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
                bin_indices = np.where(mask)[0]
                
                if len(bin_indices) > 0:
                    if len(bin_indices) <= points_per_bin:
                        selected_indices.extend(bin_indices)
                    else:
                        # Random sample from this distance bin
                        sampled = np.random.choice(bin_indices, points_per_bin, replace=False)
                        selected_indices.extend(sampled)
            
            selected_indices = np.array(selected_indices)
            print(f"[SAMPLING] Distance-based reduction: {coords.shape[0]} -> {len(selected_indices)} points")
            return coords[selected_indices], selected_indices
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    # This is the original farthest point sampling, but it's really slow
    # def farthest_point_sampling_numpy(self, coords, num_points):
    #     """
    #     Simple numpy implementation of farthest point sampling.
    #     """
    #     n_points = coords.shape[0]
    #     if num_points >= n_points:
    #         return np.arange(n_points)
        
    #     # Start with a random point
    #     selected = [np.random.randint(n_points)]
    #     distances = np.full(n_points, np.inf)
        
    #     for _ in range(1, num_points):
    #         # Update distances to nearest selected point
    #         last_selected = coords[selected[-1]]
    #         new_distances = np.linalg.norm(coords - last_selected, axis=1)
    #         distances = np.minimum(distances, new_distances)
            
    #         # Select farthest point
    #         farthest_idx = np.argmax(distances)
    #         selected.append(farthest_idx)
        
    #     return np.array(selected)
    def farthest_point_sampling_numpy(self, points, num_samples):
        points = torch.tensor(points, dtype=torch.float32).cuda()  # Send to GPU
        N = points.shape[0]
        if num_samples >= N:
            return torch.arange(N)

        selected_idx = torch.zeros(num_samples, dtype=torch.long).cuda()
        distances = torch.full((N,), float('inf')).cuda()
        farthest = torch.randint(0, N, (1,), device='cuda').item()

        for i in range(num_samples):
            selected_idx[i] = farthest
            dist = torch.norm(points - points[farthest], dim=1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances).item()

        return selected_idx.cpu().numpy()
    
    def estimate_grid_size(self, coords, target_points):
        """
        Estimate optimal grid size to achieve target number of points.
        """
        bbox_volume = np.prod(coords.max(axis=0) - coords.min(axis=0))
        target_voxel_volume = bbox_volume / target_points
        return target_voxel_volume ** (1/3)  # Cube root for 3D


class WildScenesDatasetMM(BaseDatasetMM):
    '''
    A wrapper dataset class for WildScenes multimodal data that sets up
    `WildScenesCylinderMM` instances for train, validation, and test splits.

    This class inherits from `BaseDatasetMM` and is responsible for:
    - Parsing dataset options from an OmegaConf configuration object (`dataset_opt`).
    - Initializing `WildScenesCylinderMM` for each data split (train, val, test),
      passing the appropriate configurations such as data paths, pickle directories,
      transforms, and image resolution.
    - Making the train, validation, and test datasets available as `self.train_dataset`,
      `self.val_dataset`, and `self.test_dataset` respectively.

    The configuration (`dataset_opt`) is expected to provide:
    - `dataroot`: The root path to the WildScenes dataset.
    - `split_pickle`: Path to the directory containing 3D metadata pickle files
      (e.g., `wildscenes_opt3d`).
    - `split_2droot` (optional): Path to the directory for the processed 2D
      data structure (e.g., `wildscenes_opt2d`).
    - `resolution_2d` (optional): Target image resolution (width, height),
      defaults to (1008, 756) if not specified (previously (2016, 1512) but
      reduced due to OOM issues).
    - Various transform configurations.
    '''

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.image_size = tuple(dataset_opt.get("resolution_2d", (2016, 1512))) # Tested with 1008,756, expected (2016, 1512)
        
        # Get pickle directory paths from dataset_opt
        # The main pickle (split_pickle in YAML) is for 3D data
        split_pickle_3d_dir = dataset_opt.get("split_pickle") 
        # The split_2droot in YAML is for the 2D data pickle directory
        split_pickle_2d_dir = dataset_opt.get("split_2droot", None) 

        if not split_pickle_3d_dir:
            raise ValueError("Dataset config must specify 'split_pickle' for 3D data infos.")

        self.train_dataset = WildScenesCylinderMM(
            root=self._data_path,
            split="train",
            split_pickle_3d_dir=split_pickle_3d_dir,
            split_pickle_2d_dir=split_pickle_2d_dir,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.train_transform_image,
            image_size=self.image_size
        )

        self.val_dataset = WildScenesCylinderMM(
            root=self._data_path,
            split="val",
            split_pickle_3d_dir=split_pickle_3d_dir,
            split_pickle_2d_dir=split_pickle_2d_dir,
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.val_transform_image,
            image_size=self.image_size
        )

        self.test_dataset = WildScenesCylinderMM(
            root=self._data_path,
            split="test",
            split_pickle_3d_dir=split_pickle_3d_dir,
            split_pickle_2d_dir=split_pickle_2d_dir,
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.test_transform_image,
            image_size=self.image_size
        )

        print("[INFO] WildScenesDatasetMM initialized successfully.")

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
