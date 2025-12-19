import cv2
import numpy as np
import torch
from utils.rangeimage_utils_wildcross import loadCloudFromBinary, createRangeImageWildVPR
from utils.model_utils import get_transforms

import os 
from glob import glob 
import pickle 
import random 
import pandas as pd 
from glob import glob 
from sklearn.neighbors import KDTree
from tqdm import tqdm 




def get_dataloader(mode, CFG):
    transforms = get_transforms(mode=mode, size=CFG.size)
    dataset = CLIPDataset(
        transforms=transforms,
        CFG=CFG,
        mode = mode,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

class WildVPRRangeDatasetConstruct:
    def __init__(self, data_path, sequences):
        self.data_path = data_path
        self.sequences = sequences
        print(self.sequences)
        
    def run(self):
        info = {}
        for seq in self.sequences:
            df_submaps = pd.read_csv(os.path.join(self.data_path, seq, 'submap_poses.csv'))
            df_images = pd.read_csv(os.path.join(self.data_path, seq, 'camera_poses.csv'))
            
            if 'K' in seq:
                df_submaps['x'] += 10000
            
            ts_submaps = df_submaps['%time'].to_numpy()
            coord_submaps = df_submaps[['x','y','z']].to_numpy()
            
            ts_images = df_images['%time'].to_numpy()
            coord_images = df_images[['x','y','z']].to_numpy()
            
            submap_filepaths = sorted(glob(os.path.join(self.data_path, seq, 'Clouds', '*')))
            image_filepaths = sorted(glob(os.path.join(self.data_path, seq, 'images_shrunk', '*')))
            
            assert len(df_submaps) == len(submap_filepaths), seq
            assert len(df_images) == len(image_filepaths), seq
            
            tree = KDTree(ts_images.reshape(-1,1))
            
            for anchor_idx in tqdm(np.arange(len(submap_filepaths)), desc = f"Calculating positives for {seq}"):
                anchor_ts = ts_submaps[anchor_idx]
                anchor_coord = coord_submaps[anchor_idx]
                
                pos_idx = tree.query_radius(anchor_ts.reshape(1,1), 0.5)[0]
                pos_images = [image_filepaths[pidx] for pidx in pos_idx]
                pos_coords = [coord_images[pidx] for pidx in pos_idx]
                
                if len(pos_images) == 0:
                    continue 
                
                info[len(info)] = {
                    'submap_filepath': submap_filepaths[anchor_idx],
                    'submap_coord': anchor_coord,
                    'image_filepaths': pos_images,
                    'image_coords': pos_coords
                }
        return info 

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, CFG, mode):
        self.transforms = transforms 
        self.data_path = CFG.data_path 
        self.CFG = CFG 
        assert self.CFG.query_idx is not None 
        self.get_info(mode)
        
    def get_info(self, mode):        
        train_pickle_name = os.path.join(os.path.dirname(__file__), f"wildcross_range_train_info_qidx_{self.CFG.query_idx}.pickle")
        val_pickle_name = os.path.join(os.path.dirname(__file__), f"wildcross_range_val_info_qidx_{self.CFG.query_idx}.pickle")
        if not os.path.exists(train_pickle_name) or not os.path.exists(val_pickle_name):
            sequences_venman = ['V-01','V-02','V-03','V-04']
            sequences_karawatha = ['K-01','K-02','K-03','K-04']
            val_sequences = [sequences_venman.pop(self.CFG.query_idx)]   
            sequences_karawatha.pop(self.CFG.query_idx)
            
            train_sequences = sequences_venman + sequences_karawatha    
            
            train_info = WildVPRRangeDatasetConstruct(self.data_path, train_sequences).run()
            val_info = WildVPRRangeDatasetConstruct(self.data_path, val_sequences).run()
            with open(train_pickle_name, 'wb') as f:
                pickle.dump(train_info, f)
            with open(val_pickle_name, 'wb') as f:
                pickle.dump(val_info, f)
                
        if mode == 'train': 
            self.info = pickle.load(open(train_pickle_name, 'rb'))
        else:
            self.info = pickle.load(open(val_pickle_name, 'rb'))
        
        
            
        
    def __len__(self):
        return len(self.info)
    
    def flush(self):
        pass 

    
    
    def __getitem__(self, idx):
        info_idx = self.info[idx]
        lidar_path = info_idx['submap_filepath']
        lidar_coord = info_idx['submap_coord']
        
        pos_image_idx = random.choice(range(len(info_idx['image_filepaths'])))
        image_path = info_idx['image_filepaths'][pos_image_idx]
        image_coord = info_idx['image_coords'][pos_image_idx]
        
        

        # Load image 
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        img = self.transforms(image=img)['image']
        img = torch.tensor(img).permute(2,0,1).float()
        
        # Load lidar
        lidar_points = loadCloudFromBinary(lidar_path)
        lidar_image = createRangeImageWildVPR(lidar_points, self.CFG.crop, self.CFG.crop_distance, self.CFG.distance_threshold)

        lidar_image = self.transforms(image=lidar_image)['image']
        lidar_image = torch.tensor(lidar_image).permute(2,0,1).float()
        

    
        return {'camera_image': img, 'lidar_image': lidar_image, 'camera_coord': image_coord, 'lidar_coord': lidar_coord}
        
        
    