import torch
import lightning as L
from torch.utils.data.dataloader import DataLoader
from data.dataset import WildCrossDataset 
from torchpack.utils.config import configs as CFG 
from torchvision.transforms import v2  as T

class WildCrossDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.wildcross_path = CFG.wildcross_path 
        self.split_idx = CFG.split_idx 
        
        self.pos_exclude_radius = CFG.pos_exclude_radius 
        self.pos_include_radius = CFG.pos_include_radius
        self.fov_include_radius = CFG.fov_include_radius 
        self.anchor_framerate_downsample = CFG.anchor_framerate_downsample 
        self.img_per_place = CFG.img_per_place 
        self.train_img_size = CFG.train_img_size 
        
        self.batch_size = CFG.batch_size 
        self.num_workers = CFG.num_workers 
        
        self.train_transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.train_img_size, interpolation=3),
            T.RandAugment(num_ops=3, magnitude=15, interpolation=2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def setup(self):
        print(self.wildcross_path)
        self.train_dataset = WildCrossDataset(
            data_root = self.wildcross_path,
            pos_exclude_radius = self.pos_exclude_radius,
            pos_include_radius = self.pos_include_radius,
            fov_include_radius = self.fov_include_radius,
            anchor_framerate_downsample = self.anchor_framerate_downsample,
            img_per_place = self.img_per_place,
            transform = self.train_transform,
            split_idx = self.split_idx 
        )
        
        self.val_datasets = []
        
    def train_dataloader(self):
        self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = True 
        )

if __name__ == '__main__':
    CFG.load('/datasets/work/d61-csirorobotics2/work/kni101/WildCross/benchmarking/VPR/configs/BoQ.yaml')
    module = WildCrossDataModule()
    dl = module.train_dataloader()
