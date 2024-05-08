import os
import json
import shutil
import numpy as np

from PIL import Image
from pathlib import Path
from src.data.utils import read_calib_file, read_depth, read_rgb, downsample_depth
from torch.utils.data import Dataset


raw_data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"


class KittiDataset(Dataset):
    def __init__(self, root_dir=raw_data_dir, load_raw: bool = True, train=True, downsample_lidar=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): True if dataset is training data, False for validation.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.load_raw = load_raw

        self.mode = 'train' if self.train else 'val'
        self.data_list = self.load_data()

        self.downsample_lidar = downsample_lidar

    def load_data(self):
        if self.load_raw:
            return self._load_from_raw()
        else:
            return self._load_from_processed()
        
    def _load_from_processed(self):
        data_path = self.root_dir + ("/train/" if self.train else "/valid/")
        with open(data_path + "data_list.json", 'r') as json_file:
            data_list = json.load(json_file)
        return [
            {key: data_path + path for key, path in example.items()} 
            for example in data_list
        ]

    def _load_from_raw(self):
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
    
    def _preprocess_dataset(self, output_dir: str, new_size: tuple):
        updated_images = []
    
        for image_info in self.data_list:
            updated_image_info = {}
            
            for key, old_path in image_info.items():
                rel_path = os.path.relpath(old_path, self.root_dir)
                new_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                
                if not old_path.endswith('.png'):
                    shutil.copy(old_path, new_path)
                else:
                    img = Image.open(old_path)
                    resized_img = img.resize(new_size)
                    resized_img.save(new_path)
                
                updated_image_info[key] = rel_path
            
            updated_images.append(updated_image_info)

        with open(output_dir + "/data_list.json", 'w') as json_file:
            json.dump(updated_images, json_file, indent=4)
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sparse = np.expand_dims(read_depth(data['sparse']), -1).transpose(2, 0 ,1)
        gt = np.expand_dims(read_depth(data['gt']), -1).transpose(2, 0 ,1)
        rgb = read_rgb(data['rgb']).transpose(2, 0 ,1)

        if self.downsample_lidar:
            sparse = downsample_depth(sparse, 1000)

        _, h1, w1 = rgb.shape
        _, h2, w2  = sparse.shape
        _, h3, w3  = gt.shape

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        return rgb, sparse, gt