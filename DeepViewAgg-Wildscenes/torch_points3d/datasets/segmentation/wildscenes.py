import os
import torch
import numpy as np
from sys import exit
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Sampler
import logging
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm as tq
from random import shuffle
from datetime import datetime
import os.path as osp
import glob
import pickle
from pathlib import Path

import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# WildScenes specific configuration
WILDSCENES_NUM_CLASSES = 13

# WildScenes class mappings (updated to match official WildScenes)
WILDSCENES_LABELS = [
    'bush',         # 0
    'dirt',         # 1
    'fence',        # 2
    'grass',        # 3
    'gravel',       # 4
    'log',          # 5
    'mud',          # 6
    'object',       # 7
    'other-terrain',# 8
    'rock',         # 9
    'structure',    # 10
    'tree-foliage', # 11
    'tree-trunk',   # 12
]

# Official WildScenes raw label to training label mapping
# Raw labels (from .label files) -> Training labels (0-12, 255 for ignored)
# This mapping has been updated to align perfectly with the 2D model's mapping, while also being based on the official mapping in /wildscenes/configs3d/_base_/datasets/wildscenes.py
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

########################################################################
#                                 Utils                                #
########################################################################

def read_wildscenes_bin_data(bin_path, label_path=None):
    """Read WildScenes .bin point cloud and .label files"""
    data = Data()
    points_loaded = False
    labels_loaded = False
    
    log.debug(f"Attempting to read BIN: {bin_path}")
    try:
        # Read .bin file - WildScenes3d uses XYZ format (3D labelled point clouds)
        raw_point_data = np.fromfile(bin_path, dtype=np.float32)
        
        # Prioritize XYZ format first (as per WildScenes documentation), then fall back to XYZI
        if raw_point_data.size % 3 == 0:
            raw_point_data = raw_point_data.reshape((-1, 3))
            data.pos = torch.from_numpy(raw_point_data[:, :3]) # XYZ
            log.debug(f"Read {data.pos.shape[0]} points (XYZ format) from {bin_path}")
        elif raw_point_data.size % 4 == 0: # Fallback to XYZI if needed
            raw_point_data = raw_point_data.reshape((-1, 4))
            data.pos = torch.from_numpy(raw_point_data[:, :3]) # XYZ from XYZI
            log.debug(f"Read {data.pos.shape[0]} points (XYZI format, using XYZ) from {bin_path}")
        else:
            log.error(f"Cannot determine point format for {bin_path}. Array size {raw_point_data.size} not divisible by 3 or 4.")
            data.pos = None # Indicate failure

        if data.pos is not None:
            points_loaded = True
        
    except Exception as e:
        log.error(f"Failed to read BIN file {bin_path}: {e}", exc_info=True)
        data.pos = None # Ensure pos is None on error

    if label_path:
        log.debug(f"Attempting to read label file: {label_path}")
        if os.path.exists(label_path):
            try:
                labels_np = np.fromfile(label_path, dtype=np.uint32)
                # Apply official WildScenes label mapping
                mapped_labels = np.vectorize(RAW2TRAIN.get)(labels_np, 255)  # Default to 255 (ignored) for unknown labels
                data.y = torch.from_numpy(mapped_labels.astype(np.int64))
                log.debug(f"Read {len(labels_np)} raw labels from {label_path}, mapped to training labels")
                labels_loaded = True
            except Exception as e:
                log.error(f"Failed to read label file {label_path}: {e}", exc_info=True)
        else:
            log.warning(f"Label file not found: {label_path}")
    else:
        log.debug("No label_path provided.")

    if points_loaded and data.pos is not None and labels_loaded:
        num_points = data.pos.shape[0]
        num_labels = data.y.shape[0]
        if num_points != num_labels:
            if num_labels > num_points:
                log.info(f"Label count ({num_labels}) > Point count ({num_points}) for {bin_path}. This is expected for preprocessed datasets. Truncating labels to match points.")
                data.y = data.y[:num_points]
                # This is the standard approach - labels beyond the point count are discarded
                # This matches the behavior expected by MMDetection3D pipelines
            else: # num_points > num_labels
                log.warning(f"Point count ({num_points}) > Label count ({num_labels}) for {bin_path}. This indicates missing labels. Discarding labels for this sample.")
                data.y = None
        else:
            log.debug(f"Point and label counts match ({num_points}) for {bin_path}.")
    elif points_loaded and data.pos is not None and not labels_loaded:
        log.debug(f"Points loaded for {bin_path} but no labels available. This is expected for test splits or unlabeled data.")
    
    if data.pos is None: # If points failed to load, this sample is invalid
        return Data() # Return empty Data object

    return data

########################################################################
#                               Dataset                                #
########################################################################

class WildScenesCylinder(InMemoryDataset):
    """
    WildScenes dataset supporting cylindrical sampling similar to KITTI-360.
    
    Parameters
    ----------
    root : str
        Path to the WildScenes dataset root
    split : {'train', 'val', 'test'}, optional
        Dataset split to use
    sample_per_epoch : int, optional
        Number of samples per epoch for random sampling
    radius : float, optional
        Radius of cylindrical samples
    sample_res : float, optional
        Sampling resolution for cylinder centers
    transform : callable, optional
        Transform function
    pre_transform : callable, optional  
        Pre-transform function
    pre_filter : callable, optional
        Pre-filter function
    file_infos : list, optional
        List of file information dictionaries
    """
    
    num_classes = WILDSCENES_NUM_CLASSES
    
    def __init__(self, root, split="train", sample_per_epoch=5000, radius=6,
                 sample_res=6, transform=None, pre_transform=None, 
                 pre_filter=None, file_infos: list = None):
        
        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._sample_res = sample_res
        self._file_infos = file_infos
        
        super(WildScenesCylinder, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        # Load processed data
        self._load_data()
    
    @property
    def split(self):
        return self._split
    
    @property
    def sample_per_epoch(self):
        return self._sample_per_epoch
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def sample_res(self):
        return self._sample_res
    
    @property
    def raw_file_names(self):
        """Get list of raw .bin files for the current split."""
        if self._file_infos is not None:
            raw_paths = []
            for info in self._file_infos:
                path_pc = None
                if isinstance(info.get('lidar_points'), dict) and 'lidar_path' in info['lidar_points']:
                    path_pc = info['lidar_points']['lidar_path']
                elif 'path_pointcloud' in info:
                    path_pc = info['path_pointcloud']
                
                if path_pc:
                    raw_paths.append(path_pc)
                else:
                    log.warning(f"[{self.split.upper()}_SPLIT] Could not find point cloud path in file_info item: {info}")
            
            log.info(f"[{self.split.upper()}_SPLIT] Using {len(raw_paths)} point cloud paths from provided file_infos.")
            if not raw_paths and self._file_infos:
                 log.warning(f"[{self.split.upper()}_SPLIT] file_infos provided but no usable point cloud paths found in items.")
            return sorted(raw_paths)

        log.info(f"[{self.split.upper()}_SPLIT] No file_infos provided. Attempting to find raw .bin files by scanning. Root: {self.root}")
        # Base directory for WildScenes3d sequences (contains Clouds and Labels)
        base_3d_dir = osp.join(self.root, "WildScenes3d")
        log.info(f"[{self.split.upper()}_SPLIT] Base directory for 3D sequences: {base_3d_dir}. Exists: {osp.exists(base_3d_dir)}")

        if not osp.exists(base_3d_dir):
            log.error(f"[{self.split.upper()}_SPLIT] CRITICAL: Base 3D directory {base_3d_dir} does not exist.")
            return []

        available_user_sequences = ["V-02", "K-03", "V-01", "V-03", "K-01"]
        defined_train_seqs = ["K-01", "K-03", "V-01", "V-02"]
        defined_val_seqs = ["V-03"]
        defined_test_seqs_placeholder = ["V-03"]

        sequences_to_scan_for_current_split = []
        if self.split == "train":
            sequences_to_scan_for_current_split = defined_train_seqs
        elif self.split == "val":
            sequences_to_scan_for_current_split = defined_val_seqs
        elif self.split == "test":
            sequences_to_scan_for_current_split = defined_test_seqs_placeholder
            log.warning(f"[{self.split.upper()}_SPLIT] Test split is using VAL sequences: {defined_test_seqs_placeholder}")

        log.info(f"[{self.split.upper()}_SPLIT] Defined sequences for this split: {sequences_to_scan_for_current_split}")

        valid_sequence_dirs = []
        for seq_name_candidate in sequences_to_scan_for_current_split:
            potential_seq_dir = osp.join(base_3d_dir, seq_name_candidate)
            if osp.isdir(potential_seq_dir):
                if seq_name_candidate in available_user_sequences:
                    valid_sequence_dirs.append(potential_seq_dir)
                    log.info(f"[{self.split.upper()}_SPLIT] Found valid and available sequence directory: {potential_seq_dir}")
                else:
                    log.warning(f"[{self.split.upper()}_SPLIT] Seq dir {potential_seq_dir} exists but '{seq_name_candidate}' not in available list. Skipping.")
            else:
                log.warning(f"[{self.split.upper()}_SPLIT] Defined sequence directory '{potential_seq_dir}' not found.")

        if not valid_sequence_dirs:
            log.error(f"[{self.split.upper()}_SPLIT] No valid sequence directories found in {base_3d_dir}.")
            return []

        all_bin_files_for_split = []
        for seq_dir_path in valid_sequence_dirs:
            seq_name = osp.basename(seq_dir_path)
            # Path to .bin files for this sequence
            clouds_path = osp.join(seq_dir_path, "Clouds") 
            log.info(f"[{self.split.upper()}_SPLIT] Scanning for .bin files in: {clouds_path}")
            if osp.isdir(clouds_path):
                bin_files_in_seq = glob.glob(osp.join(clouds_path, "*.bin"))
                if not bin_files_in_seq:
                    log.warning(f"[{self.split.upper()}_SPLIT] No .bin files found in {clouds_path} for sequence {seq_name}.")
                else:
                    log.info(f"[{self.split.upper()}_SPLIT] Found {len(bin_files_in_seq)} .bin files in {clouds_path} for sequence {seq_name}.")
                all_bin_files_for_split.extend(bin_files_in_seq)
            else:
                log.warning(f"[{self.split.upper()}_SPLIT] 'Clouds' directory not found: {clouds_path} for sequence {seq_name}.")
        
        if not all_bin_files_for_split:
            log.error(f"[{self.split.upper()}_SPLIT] CRITICAL: No .bin files collected for split.")
        else:
            log.info(f"[{self.split.upper()}_SPLIT] Collected {len(all_bin_files_for_split)} total .bin files.")

        return sorted(all_bin_files_for_split)
    
    @property
    def processed_file_names(self):
        return [f"wildscenes_{self.split}_processed_bin.pt"]
    
    def download(self):
        # WildScenes should be manually downloaded
        log.info("Please download WildScenes dataset manually")
        pass
    
    def process(self):
        """Process raw WildScenes data into PyTorch format"""
        log.info(f"[{self.split.upper()}_SPLIT] Starting to process data...")
        
        data_list = []
        
        if self._file_infos is not None:
            files_to_process_info = self._file_infos
            log.info(f"[{self.split.upper()}_SPLIT] Using {len(files_to_process_info)} items from provided file_infos to process.")
        else:
            # Fallback to discovering files if _file_infos is not provided
            log.info(f"[{self.split.upper()}_SPLIT] No file_infos. Discovering raw .bin files for processing.")
            discovered_bin_files = self.raw_file_names # This will trigger the scanning logic
            if not discovered_bin_files:
                log.error(f"[{self.split.upper()}_SPLIT] No raw .bin files discovered. Cannot process data.")
                os.makedirs(osp.dirname(self.processed_paths[0]), exist_ok=True)
                torch.save(data_list, self.processed_paths[0]) # Save empty list
                log.info(f"[{self.split.upper()}_SPLIT] Saved 0 processed files to {self.processed_paths[0]}.")
                return
            # Structure them like file_infos for consistent processing loop
            # For fallback, we only have bin_file, label_file will be derived
            files_to_process_info = [{'path_pointcloud': bin_f} for bin_f in discovered_bin_files]
            log.info(f"[{self.split.upper()}_SPLIT] Found {len(files_to_process_info)} raw .bin files by discovery.")

        if not files_to_process_info:
            log.error(f"[{self.split.upper()}_SPLIT] No files to process (either from file_infos or discovery). Cannot process data.")
            os.makedirs(osp.dirname(self.processed_paths[0]), exist_ok=True)
            torch.save(data_list, self.processed_paths[0])
            log.info(f"[{self.split.upper()}_SPLIT] Saved empty data list to {self.processed_paths[0]}.")
            return
            
        successfully_appended_count = 0
        for i, file_info in enumerate(tq(files_to_process_info, desc=f"Processing {self.split} split data")):
            try:
                bin_file = None
                label_file = None

                if self._file_infos is not None: # Using provided file_infos
                    # Try extracting from common nested structure first
                    if isinstance(file_info.get('lidar_points'), dict) and 'lidar_path' in file_info['lidar_points']:
                        bin_file = file_info['lidar_points']['lidar_path']
                    elif 'path_pointcloud' in file_info: # Fallback to direct key
                        bin_file = file_info['path_pointcloud']

                    if 'pts_semantic_mask_path' in file_info:
                        label_file = file_info['pts_semantic_mask_path']
                    elif 'path_labels' in file_info: # Fallback to direct key
                        label_file = file_info['path_labels']
                    
                    if not bin_file:
                        log.warning(f"[{self.split.upper()}_SPLIT_DETAIL] ({i+1}/{len(files_to_process_info)}) Missing point cloud path in file_info item: {file_info}. Skipping.")
                        continue
                    if not label_file: # When using author's pickles, we expect a label path.
                        log.warning(f"[{self.split.upper()}_SPLIT_DETAIL] ({i+1}/{len(files_to_process_info)}) Missing label path (e.g. 'pts_semantic_mask_path' or 'path_labels') in file_info for {bin_file}. Skipping sample as per author's pickle indication.")
                        continue
                else: # Fallback to discovery mode (path_pointcloud is set, derive label_file)
                    bin_file = file_info.get('path_pointcloud') # Should always be there in this branch
                    if not bin_file: # Should not happen
                        log.error(f"[{self.split.upper()}_SPLIT_DETAIL] Internal error: path_pointcloud missing in fallback mode. Skipping.")
                        continue
                    
                    # Derive label path for fallback discovery mode
                    parts = bin_file.split(os.sep)
                    if len(parts) < 4: 
                        log.error(f"BIN path {bin_file} too short for label derivation. Skipping.")
                        continue
                    seq_name = parts[-3]
                    scan_basename = os.path.basename(bin_file)
                    label_name = scan_basename.replace('.bin', '.label')
                    label_file = osp.join(self.root, "WildScenes3d", seq_name, "Labels", label_name)
                
                if i < 5: 
                    log.info(f"[{self.split.upper()}_SPLIT_DETAIL] ({i+1}/{len(files_to_process_info)}) Processing BIN: {bin_file}, Expecting Label: {label_file}")

                data = read_wildscenes_bin_data(bin_file, label_file)
                
                if data.pos is None or len(data.pos) == 0:
                    if i < 5: log.warning(f"[{self.split.upper()}_SPLIT_DETAIL] read_wildscenes_bin_data returned no points for {bin_file}. Skipping.")
                    else: log.warning(f"[{self.split.upper()}_SPLIT] No points for {bin_file}. Skipping.")
                    continue

                if i < 5:
                    log.info(f"[{self.split.upper()}_SPLIT_DETAIL] read_wildscenes_bin_data returned {data.pos.shape[0]} points.")
                    if hasattr(data, 'y') and data.y is not None:
                        log.info(f"[{self.split.upper()}_SPLIT_DETAIL] Label shape: {data.y.shape}")
                    else:
                        log.warning(f"[{self.split.upper()}_SPLIT_DETAIL] No Label data (data.y is None or missing).")

                if not hasattr(data, 'y') or data.y is None:
                    log.warning(f"[{self.split.upper()}_SPLIT] Labels missing or invalid for {bin_file}. (Expected at {label_file})")
                    if self.split == 'train':
                        log.error(f"[{self.split.upper()}_SPLIT] Skipping {bin_file} for TRAIN split due to missing/invalid labels.")
                        continue
                
                data_before_transform = None
                if self.pre_transform is not None and i < 5:
                    data_before_transform = data.clone()

                if self.pre_transform is not None:
                    if i < 5: log.info(f"[{self.split.upper()}_SPLIT_DETAIL] Applying pre_transform to {bin_file}")
                    data = self.pre_transform(data)
                    if i < 5 and data_before_transform and hasattr(data_before_transform, 'pos') and data_before_transform.pos is not None:
                        pos_len_before = len(data_before_transform.pos)
                        pos_len_after = len(data.pos) if hasattr(data, 'pos') and data.pos is not None else 'N/A'
                        log.info(f"[{self.split.upper()}_SPLIT_DETAIL] Pre_transform changed #points from {pos_len_before} to {pos_len_after}")
                    if not hasattr(data, 'pos') or data.pos is None or len(data.pos) == 0:
                        log.warning(f"[{self.split.upper()}_SPLIT] Points became empty after pre_transform for {bin_file}. Skipping.")
                        continue
                
                if self.pre_filter is None:
                    if i < 5: log.info(f"[{self.split.upper()}_SPLIT_DETAIL] No pre_filter. Appending data for {bin_file}.")
                    data_list.append(data)
                    successfully_appended_count +=1
                elif self.pre_filter(data):
                    if i < 5: log.info(f"[{self.split.upper()}_SPLIT_DETAIL] pre_filter PASSED for {bin_file}. Appending data.")
                    data_list.append(data)
                    successfully_appended_count +=1
                else:
                    if i < 5: log.warning(f"[{self.split.upper()}_SPLIT_DETAIL] {bin_file} filtered out by pre_filter.")
                    else: log.info(f"[{self.split.upper()}_SPLIT] {bin_file} filtered out by pre_filter.")
                    
            except Exception as e:
                log.error(f"[{self.split.upper()}_SPLIT] Exception during processing of {bin_file} (label: {label_file}). Error: {e}", exc_info=True)
                if i < 5:
                    log.error(f"[{self.split.upper()}_SPLIT_DETAIL] Exception for {bin_file}. See main error log above.", exc_info=True)
                continue
        
        log.info(f"[{self.split.upper()}_SPLIT] Processing loop finished. Appended {successfully_appended_count} items. Data_list size: {len(data_list)}.")
        if not data_list:
            log.error(f"[{self.split.upper()}_SPLIT] Data list empty after processing {len(files_to_process_info)} files.")
        
        # Collate and save the data in the format expected by InMemoryDataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        log.info(f"[{self.split.upper()}_SPLIT] Saved {len(data_list)} items to {self.processed_paths[0]}.")
    
    def _load_data(self):
        """Load processed data."""
        force_reprocess_debug = False 
        if force_reprocess_debug or not osp.exists(self.processed_paths[0]):
            if force_reprocess_debug:
                log.warning(f"[{self.split.upper()}_SPLIT] DEBUG: Forcing re-processing.")
            else:
                log.info("Processed data not found, processing...")
            self.process()
        
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            log.info(f"Loaded {self.len()} samples for {self.split} from {self.processed_paths[0]}.")
        except Exception as e:
            log.error(f"Failed to load processed file {self.processed_paths[0]}: {e}", exc_info=True)
            self.data, self.slices = None, None

    def __len__(self):
        """Number of samples in the dataset."""
        if self.split == 'train' and self.sample_per_epoch > 0:
            return self.sample_per_epoch
        return super().__len__()

    def __getitem__(self, idx):
        """Get a data sample"""
        if self.split == 'train' and self.sample_per_epoch > 0:
            # Randomly sample for training
            idx = np.random.randint(0, super().__len__())
        
        # Get the original data object
        data = self.get(idx)

        # Create a copy to be augmented by transforms
        data = data.clone()

        # Apply cylindrical sampling
        if self.radius > 0:
            data = self._apply_cylindrical_sampling(data)

        # Apply transforms
        if self.transform:
            data = self.transform(data)

        return data

    def _apply_cylindrical_sampling(self, data):
        """Helper to sample a cylinder from a point cloud."""
        if len(data.pos) == 0:
            return data

        # Pick a random center point from the point cloud
        center_idx = np.random.randint(0, len(data.pos))
        center = data.pos[center_idx]
        
        # Calculate distances from center (only XY plane for cylindrical)
        distances = torch.norm(data.pos[:, :2] - center[:2], dim=1)
        
        # Keep points within radius
        mask = distances <= self.radius
        
        # Apply mask to all attributes that have the same number of points
        num_points = data.num_nodes
        for key, value in data:
            if torch.is_tensor(value) and value.size(0) == num_points:
                data[key] = value[mask]
        
        return data


class WildScenesDataset(BaseDataset):
    """Main WildScenes dataset class wrapping train/val/test splits"""
    
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        
        # Extract sampling parameters
        sample_per_epoch = dataset_opt.get('sample_per_epoch', 5000)
        radius = dataset_opt.get('radius', 6)
        sample_res = dataset_opt.get('eval_sample_res', radius)

        file_infos_train = None
        file_infos_val = None
        file_infos_test = None

        author_pickle_base_path_str = dataset_opt.get("author_pickle_base_path", None)
        if author_pickle_base_path_str:
            log.info(f"Attempting to load author-defined splits from pickle base path: {author_pickle_base_path_str}")
            author_pickle_base_path = Path(author_pickle_base_path_str)
            
            pickle_path_train = author_pickle_base_path / "wildscenes_infos_train.pkl"
            pickle_path_val = author_pickle_base_path / "wildscenes_infos_val.pkl"
            pickle_path_test = author_pickle_base_path / "wildscenes_infos_test.pkl"

            try:
                if pickle_path_train.exists():
                    with open(pickle_path_train, "rb") as f:
                        # The pickle might contain a dict with 'data_list' or be the list itself
                        loaded_data = pickle.load(f)
                        if isinstance(loaded_data, dict) and 'data_list' in loaded_data:
                            file_infos_train = loaded_data['data_list']
                        elif isinstance(loaded_data, list):
                            file_infos_train = loaded_data
                        else:
                            log.warning(f"Unexpected data type in {pickle_path_train}. Expected list or dict with 'data_list'.")
                    log.info(f"Loaded {len(file_infos_train) if file_infos_train else 0} file infos for TRAIN split from {pickle_path_train}")
                else:
                    log.warning(f"TRAIN pickle not found at {pickle_path_train}")

                if pickle_path_val.exists():
                    with open(pickle_path_val, "rb") as f:
                        loaded_data = pickle.load(f)
                        if isinstance(loaded_data, dict) and 'data_list' in loaded_data:
                            file_infos_val = loaded_data['data_list']
                        elif isinstance(loaded_data, list):
                            file_infos_val = loaded_data
                        else:
                            log.warning(f"Unexpected data type in {pickle_path_val}. Expected list or dict with 'data_list'.")
                    log.info(f"Loaded {len(file_infos_val) if file_infos_val else 0} file infos for VAL split from {pickle_path_val}")
                else:
                    log.warning(f"VAL pickle not found at {pickle_path_val}")

                if pickle_path_test.exists():
                    with open(pickle_path_test, "rb") as f:
                        loaded_data = pickle.load(f)
                        if isinstance(loaded_data, dict) and 'data_list' in loaded_data:
                            file_infos_test = loaded_data['data_list']
                        elif isinstance(loaded_data, list):
                            file_infos_test = loaded_data
                        else:
                            log.warning(f"Unexpected data type in {pickle_path_test}. Expected list or dict with 'data_list'.")
                    log.info(f"Loaded {len(file_infos_test) if file_infos_test else 0} file infos for TEST split from {pickle_path_test}")
                else:
                    log.warning(f"TEST pickle not found at {pickle_path_test}")
            except Exception as e:
                log.error(f"Error loading author pickles from {author_pickle_base_path_str}: {e}", exc_info=True)
        else:
            log.info("No 'author_pickle_base_path' provided in dataset_opt. WildScenesCylinder will scan for files.")
        
        # Create train/val/test datasets
        self.train_dataset = WildScenesCylinder(
            root=self.dataset_opt.dataroot,
            split="train",
            sample_per_epoch=sample_per_epoch,
            radius=radius,
            sample_res=sample_res,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            file_infos=file_infos_train # Pass train file_infos
        )
        
        self.val_dataset = WildScenesCylinder(
            root=self.dataset_opt.dataroot,
            split="val", 
            sample_per_epoch=0,  # No random sampling for val
            radius=radius,
            sample_res=sample_res,
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            file_infos=file_infos_val # Pass val file_infos
        )
        
        self.test_dataset = WildScenesCylinder(
            root=self.dataset_opt.dataroot,
            split="test",
            sample_per_epoch=0,  # No random sampling for test
            radius=radius,
            sample_res=sample_res,
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            file_infos=file_infos_test # Pass test file_infos
        )
    
    @property 
    def class_names(self):
        return WILDSCENES_LABELS
    
    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Get tracker for monitoring training"""
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log) 