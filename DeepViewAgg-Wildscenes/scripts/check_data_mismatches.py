#!/usr/bin/env python
'''
This script checks for mismatches between point clouds and images in the WildScenes dataset.
'''

import os
import pickle
from pathlib import Path
import argparse
from collections import defaultdict

def check_data_mismatches(split_pickle_3d_dir, split_pickle_2d_dir=None, split="test"):
    """
    Check for mismatches between point clouds and images in the WildScenes dataset.
    
    Args:
        split_pickle_3d_dir (str): Path to the directory containing the 3D pickle files
        split_pickle_2d_dir (str, optional): Path to the directory containing the 2D data structure
        split (str): Dataset split to check ("train", "val", or "test")
    """
    # Load 3D metadata
    pkl_3d_path = Path(split_pickle_3d_dir) / f"wildscenes_infos_{split}.pkl"
    if not pkl_3d_path.exists():
        raise FileNotFoundError(f"3D pickle file not found: {pkl_3d_path}")
    
    with open(pkl_3d_path, "rb") as f:
        meta_3d = pickle.load(f)
    
    # Initialize counters
    total_samples = len(meta_3d["data_list"])
    missing_images = []
    missing_point_clouds = []
    missing_labels = []
    
    # Check each sample
    for f_3d in meta_3d["data_list"]:
        sample_id = f_3d["sample_id"]
        cloud_path = Path(f_3d["lidar_points"]["lidar_path"])
        label_path = Path(f_3d["pts_semantic_mask_path"])
        
        # Convert dots to dashes in the filename for image path
        img_name = f"{sample_id.replace('.', '-')}.png"
        
        # Try to find image in MMSegmentation structure first
        img_path = None
        if split_pickle_2d_dir:
            img_path = Path(split_pickle_2d_dir) / split / "image" / img_name
            if not img_path.exists() and not img_path.is_symlink():
                img_path = None
        
        # If not found in MMSeg structure, try original structure
        if img_path is None:
            seq = cloud_path.parent.parent.name  # e.g. 'K-03'
            img_path = Path(cloud_path).parent.parent.parent / "WildScenes2d" / seq / "image" / img_name
        
        # Check if files exist
        if not cloud_path.exists():
            missing_point_clouds.append(sample_id)
        
        if not label_path.exists():
            missing_labels.append(sample_id)
            
        if not img_path.exists() and not img_path.is_symlink():
            missing_images.append(sample_id)
    
    # Print statistics
    print(f"\nStatistics for {split} split:")
    print(f"Total samples: {total_samples}")
    print(f"Missing point clouds: {len(missing_point_clouds)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print("\nSample IDs with missing images:")
        for sample_id in missing_images:
            print(f"  {sample_id}")
    
    if missing_point_clouds:
        print("\nSample IDs with missing point clouds:")
        for sample_id in missing_point_clouds:
            print(f"  {sample_id}")
            
    if missing_labels:
        print("\nSample IDs with missing labels:")
        for sample_id in missing_labels:
            print(f"  {sample_id}")

def main():
    parser = argparse.ArgumentParser(description="Check for mismatches between point clouds and images in WildScenes dataset")
    parser.add_argument("--split_pickle_3d_dir", required=True, help="Path to directory containing 3D pickle files")
    parser.add_argument("--split_pickle_2d_dir", help="Path to directory containing 2D data structure")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to check")
    
    args = parser.parse_args()
    check_data_mismatches(args.split_pickle_3d_dir, args.split_pickle_2d_dir, args.split)

if __name__ == "__main__":
    main() 