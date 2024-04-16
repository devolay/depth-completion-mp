import os
import numpy as np
import torch

from pathlib import Path
from src.data.utils import depth_read
from torch.utils.data import Dataset
from PIL import Image

raw_data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"


class LIDARDataset(Dataset):
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
        self.subset = 'train' if self.train else 'val'
        self.data_list = self._load_image_paths_simple()

    def _load_image_paths_simple(self):
        """Load the file paths of images for LIDAR data from left and right cameras."""
        paths = []
        gt_path = os.path.join(self.root_dir, 'data_depth_annotated', self.subset)
        sparse_path = os.path.join(self.root_dir, 'data_depth_velodyne', self.subset)
        for sequence_dir in os.listdir(gt_path):
            gt_seq = os.path.join(gt_path, sequence_dir, "proj_depth", "groundtruth")
            sparse_seq = os.path.join(sparse_path, sequence_dir, "proj_depth", "velodyne_raw")
            for camera in ['image_02', 'image_03']:
                images = [
                    {
                        "sparse": os.path.join(sparse_seq, camera, image.name),
                        "groundtruth": os.path.join(gt_seq, camera, image.name)
                    }
                    for image in Path(os.path.join(gt_seq, camera)).rglob('*.png')]
                paths.extend(images)
        return paths

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        images = self.data_list[idx]
        sparse_depth_img = depth_read(images['sparse'])
        gt_depth_img = depth_read(images['groundtruth'])
        return sparse_depth_img, gt_depth_img