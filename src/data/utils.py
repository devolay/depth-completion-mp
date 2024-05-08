from PIL import Image
import numpy as np
import os


def read_rgb(file_name: str) -> np.ndarray:
    """Loads an RGB image from a file as a numpy array."""
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_rgb = np.array(Image.open(file_name).convert('RGB'))
    image_rgb = image_rgb.astype(np.float32) / 255.0
    return image_rgb

def read_depth(file_name: str) -> np.ndarray:
    """Loads depth map from 16 bits png file as a numpy array"""
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 255.0
    return image_depth


def read_calib_file(filepath: str) -> np.ndarray:
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def downsample_depth(depth: np.ndarray, num_samples: int) -> np.ndarray:
    """Downsample depth representations"""
    mask_keep = depth > 0
    n_keep = np.count_nonzero(mask_keep)

    if n_keep == 0:
        return mask_keep
    else:
        depth_sampled = np.zeros(depth.shape)
        prob = float(num_samples) / n_keep
        mask_keep =  np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
        depth_sampled[mask_keep] = depth[mask_keep]
        return depth_sampled