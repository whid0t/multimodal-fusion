import os
from PIL import Image
import numpy as np

# This map is derived from kitti360scripts/helpers/labels.py
# It maps original ID to trainID
ID_TO_TRAIN_ID_MAP = {
    0: 255,  # unlabeled
    1: 255,  # ego vehicle
    2: 255,  # rectification border
    3: 255,  # out of roi
    4: 255,  # static
    5: 255,  # dynamic
    6: 255,  # ground
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255,  # parking
    10: 255, # rail track
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 255, # guard rail
    15: 255, # bridge
    16: 255, # tunnel
    17: 5,   # pole
    18: 255, # polegroup
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, # caravan
    30: 255, # trailer
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    34: 2,   # garage (trainId 2, same as building)
    35: 4,   # gate (trainId 4, same as fence)
    36: 255, # stop
    37: 5,   # smallpole (trainId 5, same as pole)
    38: 255, # lamp
    39: 255, # trash bin
    40: 255, # vending machine
    41: 255, # box
    42: 255, # unknown construction
    43: 255, # unknown vehicle
    44: 255, # unknown object
    -1: 255, # license plate (mapping to 255 as a safe ignore value)
}
DEFAULT_IGNORE_LABEL = 255

def convert_id_to_train_id(id_image_path, output_dir):
    """
    Converts a label image from original IDs to trainIDs.

    Args:
        id_image_path (str): Path to the input label image (contains original IDs).
        output_dir (str): Base directory to save the converted trainID image.
                           The relative path from the input_root will be preserved.
    """
    try:
        # Open the original ID image
        id_img = Image.open(id_image_path)
        # Ensure it's single channel (grayscale)
        # KITTI-360 labels are typically like this.
        id_img_l = id_img.convert('L')
        id_array = np.array(id_img_l, dtype=np.uint8)

        # Create an empty array for the trainID image
        train_id_array = np.full(id_array.shape, DEFAULT_IGNORE_LABEL, dtype=np.uint8)

        # Perform the mapping
        for original_id, train_id in ID_TO_TRAIN_ID_MAP.items():
            train_id_array[id_array == original_id] = train_id
        
        # Handle any original IDs that were not in our map (should ideally not happen)
        # by ensuring they are set to the default ignore label.
        # This step is somewhat redundant if all IDs in images are keys in ID_TO_TRAIN_ID_MAP
        # or if unmapped IDs should indeed be DEFAULT_IGNORE_LABEL.
        # However, it's a good safety measure.
        # Create a mask of all known original IDs
        known_ids_mask = np.zeros_like(id_array, dtype=bool)
        for original_id in ID_TO_TRAIN_ID_MAP.keys():
            if original_id >= 0: # Assuming non-negative IDs are actual labels
                 known_ids_mask[id_array == original_id] = True
        
        # Pixels not corresponding to any known original_id should be ignore_label
        # This is mostly covered by initializing train_id_array with DEFAULT_IGNORE_LABEL
        # and then filling in specific train_ids.
        # What this ensures is if an ID like '50' (not in map) was in id_array,
        # it remains DEFAULT_IGNORE_LABEL in train_id_array.

        # Construct the output path
        # This part assumes `id_image_path` includes the original root path
        # that you want to strip. For simplicity, we'll take it from the user later.
        relative_path = os.path.relpath(id_image_path, input_root_dir) # input_root_dir will be defined in main
        output_path = os.path.join(output_dir, relative_path)

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the new trainID image
        train_id_img = Image.fromarray(train_id_array)
        train_id_img.save(output_path)
        print(f"Converted: {id_image_path} -> {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found {id_image_path}")
    except Exception as e:
        print(f"Error converting {id_image_path}: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Set these paths correctly!
    # Example: /home/jovyan/DeepViewAgg/dataset/2d/data_2d_semantics/train
    input_root_dir = input("Enter the absolute path to the input directory containing original ID label images: ")
    # Example: /home/jovyan/DeepViewAgg/dataset/2d/data_2d_semantics_trainId/train
    output_root_dir = input("Enter the absolute path to the output directory to save trainID label images: ")
    # --- End Configuration ---

    if not os.path.isdir(input_root_dir):
        print(f"Error: Input directory not found or is not a directory: {input_root_dir}")
        exit()

    if not os.path.isdir(output_root_dir):
        print(f"Output directory {output_root_dir} does not exist. Creating it.")
        os.makedirs(output_root_dir, exist_ok=True)

    print(f"Starting conversion from {input_root_dir} to {output_root_dir}")

    # Walk through the input directory
    for subdir, dirs, files in os.walk(input_root_dir):
        for filename in files:
            if filename.lower().endswith('.png'): # Process only PNG files
                id_image_path = os.path.join(subdir, filename)
                # The convert_id_to_train_id function will calculate the output path
                # based on input_root_dir and output_root_dir
                convert_id_to_train_id(id_image_path, output_root_dir)

    print("Conversion complete.")
    print(f"Converted labels saved in: {output_root_dir}")
    print("IMPORTANT: Remember to update your dataset configuration in your mmsegmentation .py file")
    print("to point to this new directory with trainID labels for both training and validation!")
