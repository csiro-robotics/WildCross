import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from glob import glob 
import os 
import numpy as np 
import PIL.Image as Image 



class WildCross(Dataset):
    def __init__(self, data_root, mode, size=(518,518), return_raw_img=False):
        
        self.mode = mode 
        self.size = size 
        self.data_root = data_root
        self.return_raw_image=return_raw_img
        
        if mode == 'train':
            sequences = ["V-03","V-04","K-02","K-03","K-04"]
        elif mode == 'val':
            sequences = ["K-02"]
        elif mode == 'eval':
            sequences = ["V-01", "K-01"]
        elif mode == 'extract':
            sequences = ["V-04"]
        
        # Get file lists
        self.rgb_filelist = []
        self.depth_filelist = []
        
        for seq in sequences:
            rgb_files = sorted(glob(os.path.join(self.data_root, seq, "images", "*")))[900:]
            depth_files = sorted(glob(os.path.join(self.data_root, seq, "depth", "*")))[900:]
            self.rgb_filelist += rgb_files
            self.depth_filelist += depth_files
        
        
        
        if mode == 'val':
            self.rgb_filelist = self.rgb_filelist[::15]
            self.depth_filelist = self.depth_filelist[::15]
        
        if mode == 'extract':
            basenames = [os.path.basename(x) for x in self.rgb_filelist]
            start_idx = 1900
            self.rgb_filelist = self.rgb_filelist[start_idx:start_idx + 15*15]
            self.depth_filelist = self.depth_filelist[start_idx:start_idx + 15*15]
        
        assert len(self.rgb_filelist) == len(self.depth_filelist)
        
        # Get transform
        net_w, net_h = size
        
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
    def __len__(self):
        return len(self.rgb_filelist)
        
    def read_depth_file(self, depth_path):
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        return depth_mm
        
    def __getitem__(self, idx):
        rgb_path = self.rgb_filelist[idx]
        depth_path = self.depth_filelist[idx]
        
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = self.read_depth_file(depth_path) / 1000.0
        
        sample = self.transform({'image': image, 'depth': depth})
        sample['valid_mask'] =  sample['depth']  > 1e-8
        sample['image_path'] = rgb_path
        if self.return_raw_image:
            sample['raw_image'] = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            sample['raw_depth'] = self.read_depth_file(depth_path.replace('depth_shrunk', 'depth')) / 1000.0 # Convert from mm to m
                
        return sample 
