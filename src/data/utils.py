from PIL import Image
import numpy as np
import os


def read_rgb(file_name):
    """Loads an RGB image from a file as a numpy array."""
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_rgb = np.array(Image.open(file_name))
    image_rgb = image_rgb.astype(np.float32) / 255.0
    return image_rgb

def read_depth(file_name):
    """Loads depth map from 16 bits png file as a numpy array"""
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 255.0
    return image_depth


def read_calib_file(filepath):
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