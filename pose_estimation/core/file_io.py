# core/file_io.py
import os
import cv2
import numpy as np

def prepare_directories(output_root: str, variant_name: str) -> tuple:
    """Creates and returns the dataset hierarchy for a specific variant."""
    rgb_dir = os.path.join(output_root, variant_name, "rgb")
    depth_dir = os.path.join(output_root, variant_name, "depth")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    return rgb_dir, depth_dir

def save_data_pair(rgb_frame, depth_data, output_root: str, variant_name: str, batch_number: str, capture_count: int) -> tuple:
    """
    Saves synchronized frames directly with batch tracking naming parameters.
    RGB: Lossless 8-bit .png | Depth: Raw 16-bit .npy
    Returns (success_bool, tuple of saved file paths)
    """
    rgb_dir, depth_dir = prepare_directories(output_root, variant_name)
    
    # Updated file-naming convention to isolate separate runs
    rgb_filename = f"batch_{batch_number}_{capture_count:04d}.png"
    depth_filename = f"batch_{batch_number}_{capture_count:04d}.npy"
    
    rgb_path = os.path.join(rgb_dir, rgb_filename)
    depth_path = os.path.join(depth_dir, depth_filename)
    
    try:
        cv2.imwrite(rgb_path, rgb_frame)
        np.save(depth_path, depth_data)
        return True, (rgb_path, depth_path)
    except Exception as e:
        print(f"I/O Error saving pair {capture_count}: {e}")
        return False, ()
