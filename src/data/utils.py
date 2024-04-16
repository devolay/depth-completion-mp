from PIL import Image
import numpy as np

def depth_read(filename: str):
    # loads depth map D from png file
    # and returns it as a numpy array,
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return depth