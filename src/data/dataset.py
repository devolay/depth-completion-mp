import os
import numpy as np
import torch

from pathlib import Path
from src.data.utils import read_calib_file, read_depth, read_rgb
from torch.utils.data import Dataset
from PIL import Image

raw_data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"


class KittiDataset(Dataset):
    def __init__(self, root_dir=raw_data_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): True if dataset is training data, False for validation.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.mode = 'train' if self.train else 'val'
        self.data_list = self._load_data()

    def _load_data(self):
        """Load the file paths of images for LIDAR data from left and right cameras."""
        paths = []
        gt_path = os.path.join(self.root_dir, 'data_depth_annotated', self.mode)
        sparse_path = os.path.join(self.root_dir, 'data_depth_velodyne', self.mode)
        raw_path = os.path.join(self.root_dir, "raw_kitti_data")
        for sequence_dir in os.listdir(gt_path):
            date = '_'.join(sequence_dir.split('_')[:3])
            gt_seq = os.path.join(gt_path, sequence_dir, "proj_depth", "groundtruth")
            sparse_seq = os.path.join(sparse_path, sequence_dir, "proj_depth", "velodyne_raw")
            raw_seq = os.path.join(raw_path, date, sequence_dir)
            for camera in ['image_02', 'image_03']:
                images = [
                    {
                        "sparse": os.path.join(sparse_seq, camera, image.name),
                        "gt": os.path.join(gt_seq, camera, image.name),
                        "rgb": os.path.join(raw_seq, camera, "data", image.name),
                        'calibration': os.path.join(raw_path, date, "calib_cam_to_cam.txt")

                    }
                    for image in Path(os.path.join(gt_seq, camera)).rglob('*.png')]
                paths.extend(images)
        return paths

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sparse = read_depth(data['sparse'])
        gt = read_depth(data['gt'])
        rgb = read_rgb(data['rgb'])

        if self.mode in ['train', 'val']:
            calib = read_calib_file(data['calibration'])
            if 'image_02' in data['rgb']:
                K_cam = np.reshape(calib['P_rect_02'], (3, 4))
            elif 'image_03' in data['rgb']:
                K_cam = np.reshape(calib['P_rect_03'], (3, 4))
            K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]]
        else:
            f_calib = open(data['calibration'], 'r')
            K_cam = f_calib.readline().split(' ')
            f_calib.close()
            K = [float(K_cam[0]), float(K_cam[4]), float(K_cam[2]), float(K_cam[5])]

        w1, h1, _ = rgb.shape
        w2, h2 = sparse.shape
        w3, h3 = gt.shape

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, sparse, gt, K